"""Tests for the shared Pi JSONL RPC transport."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from kai.pi_rpc import (
    PI_RPC_STREAM_LIMIT,
    PiRpcCommandError,
    PiRpcEOFError,
    PiRpcProtocolError,
    PiRpcTimeoutError,
    PiRpcTransport,
    decode_pi_rpc_record,
    encode_pi_rpc_command,
    pi_rpc_extension_error,
    pi_rpc_is_settled,
    pi_rpc_text_delta,
    require_pi_rpc_response,
)


def _reader_with(*records: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader(limit=PI_RPC_STREAM_LIMIT)
    for record in records:
        reader.feed_data(record)
    reader.feed_eof()
    return reader


class TestPiRpcFraming:
    def test_command_is_compact_utf8_jsonl(self):
        encoded = encode_pi_rpc_command({"id": "one", "type": "prompt", "message": "héllo"})

        assert encoded.endswith(b"\n")
        assert encoded.count(b"\n") == 1
        assert json.loads(encoded) == {"id": "one", "type": "prompt", "message": "héllo"}

    def test_unicode_line_separators_are_not_record_delimiters(self):
        encoded = encode_pi_rpc_command({"type": "prompt", "message": "before\u2028middle\u2029after"})

        assert encoded.count(b"\n") == 1
        assert b"\xe2\x80\xa8" in encoded
        assert b"\xe2\x80\xa9" in encoded
        assert decode_pi_rpc_record(encoded)["message"] == "before\u2028middle\u2029after"

    @pytest.mark.parametrize("ending", [b"\n", b"\r\n"])
    def test_decode_accepts_lf_and_optional_cr(self, ending: bytes):
        assert decode_pi_rpc_record(b'{"type":"agent_settled"}' + ending) == {"type": "agent_settled"}

    @pytest.mark.parametrize(
        ("record", "match"),
        [
            (b'{"type":"agent_settled"}', "unterminated"),
            (b"\n", "empty"),
            (b"\xff\n", "invalid UTF-8"),
            (b'{"type":\n', "invalid JSON"),
            (b"[]\n", "JSON object"),
            (b"{}\n", "non-empty string 'type'"),
        ],
    )
    def test_decode_rejects_malformed_records(self, record: bytes, match: str):
        with pytest.raises(PiRpcProtocolError, match=match):
            decode_pi_rpc_record(record)

    def test_encode_requires_command_type(self):
        with pytest.raises(PiRpcProtocolError, match="command requires"):
            encode_pi_rpc_command({"id": "missing-type"})

    def test_encode_rejects_non_json_value(self):
        with pytest.raises(PiRpcProtocolError, match="not JSON serializable"):
            encode_pi_rpc_command({"type": "prompt", "message": object()})


class TestPiRpcTransport:
    @pytest.mark.asyncio
    async def test_send_writes_and_drains_one_record(self):
        writer = MagicMock()
        writer.drain = AsyncMock()
        transport = PiRpcTransport(writer, _reader_with())

        await transport.send({"id": "state-1", "type": "get_state"})

        writer.write.assert_called_once_with(b'{"id":"state-1","type":"get_state"}\n')
        writer.drain.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_maps_broken_pipe_to_eof(self):
        writer = MagicMock()
        writer.write.side_effect = BrokenPipeError("closed")
        writer.drain = AsyncMock()
        transport = PiRpcTransport(writer, _reader_with())

        with pytest.raises(PiRpcEOFError, match="could not write"):
            await transport.send({"type": "get_state"})

    @pytest.mark.asyncio
    async def test_receive_decodes_one_record(self):
        writer = MagicMock()
        writer.drain = AsyncMock()
        transport = PiRpcTransport(writer, _reader_with(b'{"type":"agent_start"}\n'))

        assert await transport.receive() == {"type": "agent_start"}

    @pytest.mark.asyncio
    async def test_receive_reports_clean_eof(self):
        writer = MagicMock()
        writer.drain = AsyncMock()
        transport = PiRpcTransport(writer, _reader_with())

        with pytest.raises(PiRpcEOFError, match="stdout closed"):
            await transport.receive()

    @pytest.mark.asyncio
    async def test_receive_reports_unterminated_eof_fragment(self):
        writer = MagicMock()
        writer.drain = AsyncMock()
        transport = PiRpcTransport(writer, _reader_with(b'{"type":"agent_start"}'))

        with pytest.raises(PiRpcProtocolError, match="unterminated"):
            await transport.receive()

    @pytest.mark.asyncio
    async def test_receive_timeout_has_typed_error(self):
        class SlowReader:
            async def readline(self) -> bytes:
                await asyncio.sleep(60)
                return b""

        writer = MagicMock()
        writer.drain = AsyncMock()
        transport = PiRpcTransport(writer, SlowReader())  # type: ignore[arg-type]

        with pytest.raises(PiRpcTimeoutError, match="timed out"):
            await transport.receive(timeout_seconds=0.001)


class TestPiRpcSemantics:
    def test_success_response_is_correlated_and_unwrapped(self):
        data = require_pi_rpc_response(
            {
                "id": "models-1",
                "type": "response",
                "command": "get_available_models",
                "success": True,
                "data": {"models": []},
            },
            request_id="models-1",
            command="get_available_models",
        )

        assert data == {"models": []}

    def test_failed_response_has_typed_error(self):
        with pytest.raises(PiRpcCommandError, match="model not found") as excinfo:
            require_pi_rpc_response(
                {
                    "id": "model-1",
                    "type": "response",
                    "command": "set_model",
                    "success": False,
                    "error": "model not found",
                },
                request_id="model-1",
                command="set_model",
            )

        assert excinfo.value.command == "set_model"
        assert excinfo.value.request_id == "model-1"

    @pytest.mark.parametrize(
        "message",
        [
            {"type": "agent_start"},
            {"id": "other", "type": "response", "command": "get_state", "success": True},
            {"id": "state-1", "type": "response", "command": "set_model", "success": True},
            {"id": "state-1", "type": "response", "command": "get_state", "success": "yes"},
        ],
    )
    def test_response_mismatch_is_protocol_error(self, message):
        with pytest.raises(PiRpcProtocolError):
            require_pi_rpc_response(message, request_id="state-1", command="get_state")

    def test_text_delta_is_extracted(self):
        assert (
            pi_rpc_text_delta(
                {
                    "type": "message_update",
                    "assistantMessageEvent": {"type": "text_delta", "delta": "hello"},
                }
            )
            == "hello"
        )

    @pytest.mark.parametrize(
        "message",
        [
            {"type": "tool_execution_update"},
            {
                "type": "message_update",
                "assistantMessageEvent": {"type": "thinking_delta", "delta": "private"},
            },
            {
                "type": "message_update",
                "assistantMessageEvent": {"type": "toolcall_delta", "delta": "arguments"},
            },
        ],
    )
    def test_non_text_events_do_not_become_assistant_text(self, message):
        assert pi_rpc_text_delta(message) is None

    def test_malformed_text_delta_is_protocol_error(self):
        with pytest.raises(PiRpcProtocolError, match="string 'delta'"):
            pi_rpc_text_delta(
                {
                    "type": "message_update",
                    "assistantMessageEvent": {"type": "text_delta", "delta": 7},
                }
            )

    def test_only_agent_settled_is_terminal(self):
        assert not pi_rpc_is_settled({"type": "turn_end"})
        assert not pi_rpc_is_settled({"type": "agent_end", "willRetry": False})
        assert pi_rpc_is_settled({"type": "agent_settled"})

    def test_extension_error_is_safely_formatted(self):
        assert (
            pi_rpc_extension_error(
                {
                    "type": "extension_error",
                    "extensionPath": "/tmp/example.ts",
                    "error": "hook failed",
                }
            )
            == "/tmp/example.ts: hook failed"
        )
        assert pi_rpc_extension_error({"type": "agent_start"}) is None
