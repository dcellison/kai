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

    def test_default_joins_multiple_completed_items_with_blank_line(self):
        """
        A single codex turn can emit multiple agent_message items
        (e.g. preamble before a tool call, summary after). The default
        behavior must surface ALL of them joined with a blank-line
        separator so review-style callers do not silently drop earlier
        findings. Mirrors the persistent backend's per-item state
        machine from PR #491.
        """
        events = [
            {"type": "item.completed", "item": {"id": "i1", "type": "agent_message", "text": "first finding"}},
            {"type": "item.completed", "item": {"id": "i2", "type": "agent_message", "text": "second finding"}},
        ]
        stream = "\n".join(json.dumps(e) for e in events) + "\n"
        assert extract_codex_text(stream) == "first finding\n\nsecond finding"

    def test_join_items_false_returns_last_completed(self):
        """
        Opt-out path for callers whose downstream contract is "exactly
        one final agent_message" (triage parsing structured JSON). With
        join_items=False, a multi-item turn collapses to the last
        completed item only - the prior helper behavior, preserved for
        callers that would have their parse corrupted by a joined
        preamble + body.
        """
        events = [
            {"type": "item.completed", "item": {"id": "i1", "type": "agent_message", "text": "first"}},
            {"type": "item.completed", "item": {"id": "i2", "type": "agent_message", "text": "second"}},
        ]
        stream = "\n".join(json.dumps(e) for e in events) + "\n"
        assert extract_codex_text(stream, join_items=False) == "second"

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
