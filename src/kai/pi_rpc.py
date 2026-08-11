"""Shared JSONL transport primitives for Pi's RPC mode.

Pi's subprocess protocol is neither JSON-RPC nor ACP.  Commands, command
responses, and asynchronous agent events all share stdout as strict JSONL.
This module owns the framing and the small set of event semantics that the
future conversational backend and one-shot reasoner must agree on.

It intentionally does not launch Pi or make ``pi`` a selectable backend.
Process lifecycle, user isolation, and Kai ``StreamEvent`` conversion belong
to those two callers.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any, Protocol

# Match the per-line headroom used by Kai's other streaming backends.  Pi
# emits tool results and final messages as single JSON records, so the default
# asyncio subprocess reader limit (64 KiB) is too small for real agent turns.
PI_RPC_STREAM_LIMIT = 16 * 1024 * 1024

PiRpcObject = dict[str, Any]
PiRpcRequestId = str | int


class PiRpcError(RuntimeError):
    """Base class for Pi RPC transport and protocol failures."""


class PiRpcEOFError(PiRpcError):
    """Raised when Pi closes its RPC stream unexpectedly."""


class PiRpcProtocolError(PiRpcError):
    """Raised when Pi emits a malformed or inconsistent RPC record."""


class PiRpcTimeoutError(PiRpcError):
    """Raised when no complete Pi RPC record arrives before the deadline."""


class PiRpcCommandError(PiRpcError):
    """Raised when Pi returns a correlated ``success: false`` response."""

    def __init__(self, command: str, request_id: PiRpcRequestId, detail: str) -> None:
        self.command = command
        self.request_id = request_id
        self.detail = detail
        super().__init__(f"Pi RPC command {command!r} ({request_id!r}) failed: {detail}")


class _PiRpcWriter(Protocol):
    """Structural subset of ``asyncio.StreamWriter`` used by subprocesses."""

    def write(self, data: bytes) -> None: ...

    async def drain(self) -> None: ...


def encode_pi_rpc_command(command: Mapping[str, Any]) -> bytes:
    """Encode one Pi command as a UTF-8 JSONL record.

    A non-empty string ``type`` is required because Pi uses that field as the
    command discriminator.  ``ensure_ascii=False`` preserves Unicode,
    including U+2028/U+2029; they are valid inside JSON strings and must not
    be mistaken for record delimiters.
    """

    command_type = command.get("type")
    if not isinstance(command_type, str) or not command_type:
        raise PiRpcProtocolError("Pi RPC command requires a non-empty string 'type'")
    try:
        body = json.dumps(dict(command), ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise PiRpcProtocolError(f"Pi RPC command is not JSON serializable: {exc}") from exc
    return body.encode("utf-8") + b"\n"


def decode_pi_rpc_record(record: bytes) -> PiRpcObject:
    """Decode one LF-terminated Pi JSONL record.

    Pi specifies LF as the only record delimiter and permits an optional CR
    immediately before it.  Rejecting an EOF fragment without LF prevents a
    truncated final JSON value from being mistaken for a complete message.
    """

    if not record.endswith(b"\n"):
        raise PiRpcProtocolError("Pi RPC stream ended with an unterminated JSONL record")
    body = record[:-1]
    if body.endswith(b"\r"):
        body = body[:-1]
    if not body:
        raise PiRpcProtocolError("Pi RPC emitted an empty JSONL record")
    try:
        decoded = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PiRpcProtocolError(f"Pi RPC emitted invalid UTF-8: {exc}") from exc
    try:
        message = json.loads(decoded)
    except json.JSONDecodeError as exc:
        raise PiRpcProtocolError(f"Pi RPC emitted invalid JSON: {exc.msg}") from exc
    if not isinstance(message, dict):
        raise PiRpcProtocolError("Pi RPC record must be a JSON object")
    message_type = message.get("type")
    if not isinstance(message_type, str) or not message_type:
        raise PiRpcProtocolError("Pi RPC record requires a non-empty string 'type'")
    return message


class PiRpcTransport:
    """Read and write strict Pi RPC records over subprocess streams."""

    def __init__(self, stdin: _PiRpcWriter, stdout: asyncio.StreamReader) -> None:
        self._stdin = stdin
        self._stdout = stdout

    async def send(self, command: Mapping[str, Any]) -> None:
        """Write and flush one command record."""

        encoded = encode_pi_rpc_command(command)
        try:
            self._stdin.write(encoded)
            await self._stdin.drain()
        except (BrokenPipeError, ConnectionError, RuntimeError) as exc:
            raise PiRpcEOFError(f"could not write to Pi RPC stdin: {exc}") from exc

    async def receive(self, *, timeout_seconds: float | None = None) -> PiRpcObject:
        """Read one complete response or event record.

        ``timeout_seconds=None`` waits indefinitely.  Callers normally pass
        their remaining turn/startup budget so a stalled Pi process cannot
        hold a Kai request forever.
        """

        try:
            if timeout_seconds is None:
                record = await self._stdout.readline()
            else:
                record = await asyncio.wait_for(self._stdout.readline(), timeout=timeout_seconds)
        except TimeoutError as exc:
            raise PiRpcTimeoutError("timed out waiting for a Pi RPC record") from exc
        except (ConnectionError, RuntimeError) as exc:
            raise PiRpcEOFError(f"could not read Pi RPC stdout: {exc}") from exc
        if not record:
            raise PiRpcEOFError("Pi RPC stdout closed")
        return decode_pi_rpc_record(record)


def require_pi_rpc_response(
    message: Mapping[str, Any],
    *,
    request_id: PiRpcRequestId,
    command: str,
) -> Any:
    """Validate and unwrap a response for one correlated command.

    Asynchronous events may appear around command responses; callers decide
    when to invoke this helper after matching the ``id``.  A response for a
    different id is a protocol error here rather than being silently accepted.
    """

    if message.get("type") != "response":
        raise PiRpcProtocolError(f"expected Pi RPC response for {command!r}, got {message.get('type')!r}")
    if message.get("id") != request_id:
        raise PiRpcProtocolError(
            f"Pi RPC response id mismatch for {command!r}: expected {request_id!r}, got {message.get('id')!r}"
        )
    if message.get("command") != command:
        raise PiRpcProtocolError(
            f"Pi RPC response command mismatch: expected {command!r}, got {message.get('command')!r}"
        )
    success = message.get("success")
    if not isinstance(success, bool):
        raise PiRpcProtocolError(f"Pi RPC response for {command!r} requires boolean 'success'")
    if not success:
        detail = message.get("error")
        if not isinstance(detail, str) or not detail:
            detail = "unknown command error"
        raise PiRpcCommandError(command, request_id, detail)
    return message.get("data")


def pi_rpc_text_delta(message: Mapping[str, Any]) -> str | None:
    """Return a visible assistant text delta, ignoring other event kinds."""

    if message.get("type") != "message_update":
        return None
    event = message.get("assistantMessageEvent")
    if not isinstance(event, dict):
        raise PiRpcProtocolError("Pi message_update requires an assistantMessageEvent object")
    if event.get("type") != "text_delta":
        return None
    delta = event.get("delta")
    if not isinstance(delta, str):
        raise PiRpcProtocolError("Pi text_delta requires a string 'delta'")
    return delta


def pi_rpc_is_settled(message: Mapping[str, Any]) -> bool:
    """Return whether Pi has reached its session-level terminal event.

    ``agent_end`` is deliberately not terminal: current Pi may still retry,
    compact, or process queued continuation messages before ``agent_settled``.
    """

    return message.get("type") == "agent_settled"


def pi_rpc_extension_error(message: Mapping[str, Any]) -> str | None:
    """Return a safe diagnostic for an ``extension_error`` event."""

    if message.get("type") != "extension_error":
        return None
    error = message.get("error")
    if not isinstance(error, str) or not error:
        error = "unknown extension error"
    extension_path = message.get("extensionPath")
    if isinstance(extension_path, str) and extension_path:
        return f"{extension_path}: {error}"
    return error
