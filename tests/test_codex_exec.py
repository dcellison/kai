"""
Tests for kai.codex_exec - shared NDJSON parser for codex one-shot
callers (triage, review, and any future codex-driven one-shot agent).

The codex exec --json schema (codex-rs/exec/src/exec_events.rs):
each event has top-level `type` discriminator; agent_message
items carry their full text in `item.text`. item.completed is
authoritative; item.updated is interim. turn.failed terminates
extraction with an empty result.

These tests originally lived in test_triage.py alongside the helper.
Moved here when the helper was promoted to a shared module so the
test file's location matches the implementation file's location;
no behavior change.
"""

import json

from kai.codex_exec import extract_codex_text


class TestExtractCodexText:
    """Unit tests for the extract_codex_text NDJSON parser."""

    def test_empty_input(self):
        """An empty stream returns the empty string."""
        assert extract_codex_text("") == ""

    def test_skips_non_json_lines(self):
        """Non-JSON lines are silently skipped."""
        valid = json.dumps({"type": "item.completed", "item": {"id": "i", "type": "agent_message", "text": "hello"}})
        stream = "not-json\n" + valid + "\n"
        assert extract_codex_text(stream) == "hello"

    def test_extracts_item_completed_agent_message(self):
        """item.completed for an agent_message returns item.text."""
        event = {"type": "item.completed", "item": {"id": "i1", "type": "agent_message", "text": "hello"}}
        stream = json.dumps(event) + "\n"
        assert extract_codex_text(stream) == "hello"

    def test_ignores_non_agent_message_items(self):
        """item.completed for non-agent_message items (e.g. reasoning) contributes nothing."""
        event = {"type": "item.completed", "item": {"id": "i1", "type": "reasoning", "text": "thinking..."}}
        stream = json.dumps(event) + "\n"
        assert extract_codex_text(stream) == ""

    def test_ignores_lifecycle_events(self):
        """thread.started, turn.started, turn.completed contribute nothing."""
        events = [
            {"type": "thread.started", "thread_id": "thr_1"},
            {"type": "turn.started"},
            {"type": "turn.completed", "usage": {"input_tokens": 0, "output_tokens": 0}},
        ]
        stream = "\n".join(json.dumps(e) for e in events) + "\n"
        assert extract_codex_text(stream) == ""

    def test_strips_outer_whitespace(self):
        """The accumulated result has leading/trailing whitespace stripped."""
        event = {"type": "item.completed", "item": {"id": "i", "type": "agent_message", "text": "  hello  "}}
        stream = json.dumps(event) + "\n"
        assert extract_codex_text(stream) == "hello"

    def test_completed_wins_over_updated(self):
        """item.completed text supersedes any earlier item.updated text."""
        events = [
            {"type": "item.updated", "item": {"id": "i", "type": "agent_message", "text": "partial"}},
            {"type": "item.updated", "item": {"id": "i", "type": "agent_message", "text": "partial more"}},
            {"type": "item.completed", "item": {"id": "i", "type": "agent_message", "text": "FINAL"}},
        ]
        stream = "\n".join(json.dumps(e) for e in events) + "\n"
        assert extract_codex_text(stream) == "FINAL"

    def test_last_completed_wins(self):
        """Multiple item.completed events (rare): last one wins."""
        events = [
            {"type": "item.completed", "item": {"id": "i1", "type": "agent_message", "text": "first"}},
            {"type": "item.completed", "item": {"id": "i2", "type": "agent_message", "text": "second"}},
        ]
        stream = "\n".join(json.dumps(e) for e in events) + "\n"
        assert extract_codex_text(stream) == "second"

    def test_updated_fallback_when_no_completed(self):
        """If only item.updated events arrive (truncated stream), use the latest one."""
        events = [
            {"type": "item.updated", "item": {"id": "i", "type": "agent_message", "text": "first"}},
            {"type": "item.updated", "item": {"id": "i", "type": "agent_message", "text": "second"}},
        ]
        stream = "\n".join(json.dumps(e) for e in events) + "\n"
        assert extract_codex_text(stream) == "second"

    def test_turn_failed_short_circuits(self):
        """A turn.failed event terminates extraction with the empty string."""
        events = [
            {"type": "item.completed", "item": {"id": "i", "type": "agent_message", "text": "ignored"}},
            {"type": "turn.failed", "error": {"message": "model unavailable"}},
        ]
        stream = "\n".join(json.dumps(e) for e in events) + "\n"
        assert extract_codex_text(stream) == ""
