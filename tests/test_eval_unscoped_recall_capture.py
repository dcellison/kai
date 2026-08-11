"""Tests for the unscoped recall-capture helpers.

Covers `kai.eval._unscoped_recall_capture`: the buffer-and-drain
machinery (`_RecallLogCapture`, `_attach_capture`, `_detach_capture`)
that intercepts `memory.recall` log lines from
`kai.memory.format_context`, and the `legacy_retrieve_hits` public
wrapper the collision-probe generator uses as its unscoped
verification gate.

Two test classes:

1. TestLogParserRoundTrip - drives a real `format_context` call with
   Mem0 mocked, asserts the per-hit `id` field round-trips through
   the parser into `compute_rank`. Confirms the log-line shape is
   the only contract the capture handler depends on.
2. TestLegacyRetrieveHits - exercises the public wrapper's three
   load-bearing seams: happy path returns hits+latency, zero/two
   captured records raise loudly, and an exception inside
   `format_context` still detaches the capture handler (the
   try/finally invariant that motivates the helper's existence).
"""

from __future__ import annotations

import asyncio
import json
import logging
from unittest.mock import MagicMock

import pytest

from kai.config import Config
from kai.eval._probes import compute_rank
from kai.eval._unscoped_recall_capture import (
    _attach_capture,
    _detach_capture,
    legacy_retrieve_hits,
)

# Minimal Config the recall path needs: memory enabled with a
# production-shape budget and floor. The capture tests do not
# exercise sweep knobs, so only the fields format_context reads
# need to be populated.
_BASE_CONFIG = Config(
    telegram_bot_token="test-token",
    allowed_user_ids={12345},
    memory_enabled=True,
    memory_search_limit=10,
    memory_token_budget=2000,
    memory_search_floor=0.3,
)


@pytest.fixture(autouse=True)
def _clean_memory_state():
    """Reset kai.memory module-level state between tests.

    TestLogParserRoundTrip writes a MagicMock into `mem_mod._memory`
    and a stub Config into `mem_mod._config`; leaving those in
    place would silently couple later tests (here or in sibling
    test files) to this file's fixtures.
    """
    import kai.memory as mem_mod

    mem_mod._memory = None
    mem_mod._config = None
    yield
    mem_mod._memory = None
    mem_mod._config = None


# ── Test 1: Log parser end-to-end (round-trip on `id` field) ──────


class TestLogParserRoundTrip:
    """Capture a real memory.recall record, parse it, assert id passthrough.

    Downstream ranking logic depends on each per-hit dict in the
    recall payload carrying `id` matching MemoryResult.id. This
    test wires the capture handler to a real format_context call
    (with a mocked Mem0 search) and confirms the id flows through
    end-to-end via the structured log line, not via any
    side-channel.
    """

    def test_capture_and_parse_id_field(self):
        import kai.memory as mem_mod
        from kai.memory import format_context

        # Mock Mem0 to return two rows with known IDs. The capture
        # handler reads format_context's emit; the parsed payload
        # must carry both IDs so a downstream `compute_rank` against
        # expected_fact_id="row-b" returns 2 (not None).
        mock_mem = MagicMock()
        mock_mem.search.return_value = {
            "results": [
                {
                    "id": "row-a",
                    "memory": "Fact A text",
                    "score": 0.9,
                    "metadata": {"type": "fact", "source": "extracted"},
                    "created_at": "2026-04-01T10:00:00",
                },
                {
                    "id": "row-b",
                    "memory": "Fact B text",
                    "score": 0.8,
                    "metadata": {"type": "fact", "source": "extracted"},
                    "created_at": "2026-04-02T10:00:00",
                },
            ]
        }
        mem_mod._memory = mock_mem
        mem_mod._config = _BASE_CONFIG

        # Attach the capture handler exactly the way the production
        # code path does. _attach_capture also forces the kai.memory
        # logger level to INFO if it was higher (test runs default
        # to WARNING via pytest config), so the recall record is
        # guaranteed to reach the handler.
        capture = _attach_capture()
        try:
            asyncio.run(format_context("any query", user_id="42"))
            payloads = capture.drain()
        finally:
            _detach_capture(capture)

        assert len(payloads) == 1
        payload = payloads[0]
        ids = [h["id"] for h in payload["hits"]]
        # IDs round-tripped via JSON; compute_rank must locate row-b
        # at position 2 to score the prerequisite-dependent contract.
        assert ids == ["row-a", "row-b"]
        assert compute_rank(payload["hits"], "row-b") == 2
        assert compute_rank(payload["hits"], "missing-id") is None


# ── Test 2: legacy_retrieve_hits public helper ─────────────────────


class TestLegacyRetrieveHits:
    """The public wrapper around the attach/format/drain/detach dance.

    `legacy_retrieve_hits` is the helper external callers (the
    collision-probe generator in `kai.eval.gen_collision_probes`)
    use to drive the unscoped recall pipeline as a verification
    gate. The contract is: one `memory.recall` line per call
    returns `(hits, latency_ms)`; zero or more than one raises;
    the capture handler MUST be detached on every exit path so a
    long-lived process running successive probes does not silently
    double-capture later records.
    """

    @staticmethod
    def _emit_recall(payload: dict) -> None:
        """Emit a single memory.recall log line.

        Mirrors what `kai.memory.format_context` does internally
        so tests can drive the capture without standing up Mem0.
        The prefix and JSON-payload shape are the parser's
        contract; if either changes, `_RecallLogCapture.drain`
        breaks first and every capture test fails together, so we
        encode the prefix inline rather than importing the private
        constant.
        """
        logging.getLogger("kai.memory").info("memory.recall " + json.dumps(payload))

    def test_returns_hits_and_latency_from_single_payload(self, monkeypatch):
        """Happy path: one captured payload returns its hits and latency."""

        async def fake_format_context(query: str, *, user_id: str, token_budget=None):
            self._emit_recall(
                {
                    "user_id": user_id,
                    "query": query,
                    "hits": [
                        {"id": "row-a", "score": 0.9},
                        {"id": "row-b", "score": 0.7},
                    ],
                    "latency_ms": 42,
                    "lines_used": 2,
                }
            )

        # `legacy_retrieve_hits` does a deferred `from kai.memory
        # import format_context` inside the function body, so the
        # patch must target the kai.memory attribute that the
        # imported name resolves to, not the eval module.
        monkeypatch.setattr("kai.memory.format_context", fake_format_context)

        hits, latency_ms = asyncio.run(legacy_retrieve_hits("any query", user_id="42"))

        assert latency_ms == 42
        assert [h["id"] for h in hits] == ["row-a", "row-b"]

    def test_raises_when_zero_recall_lines_captured(self, monkeypatch):
        """A format_context that emits no memory.recall line is a logging-discipline regression.

        Picking "zero hits" silently would make every downstream
        score wrong in ways no other test catches; raising loudly
        is the documented contract and the public helper inherits
        it.
        """

        async def fake_format_context(query: str, *, user_id: str, token_budget=None):
            # Deliberately no log emit. format_context returning
            # nothing must surface as RuntimeError, not as an empty
            # hits list.
            return None

        monkeypatch.setattr("kai.memory.format_context", fake_format_context)

        with pytest.raises(RuntimeError, match="got 0"):
            asyncio.run(legacy_retrieve_hits("any query", user_id="42"))

    def test_raises_when_multiple_recall_lines_captured(self, monkeypatch):
        """Two emits in one call indicates a logging duplication regression."""

        async def fake_format_context(query: str, *, user_id: str, token_budget=None):
            self._emit_recall({"hits": [], "latency_ms": 1, "user_id": user_id, "query": query})
            self._emit_recall({"hits": [], "latency_ms": 2, "user_id": user_id, "query": query})

        monkeypatch.setattr("kai.memory.format_context", fake_format_context)

        with pytest.raises(RuntimeError, match="got 2"):
            asyncio.run(legacy_retrieve_hits("any query", user_id="42"))

    def test_detaches_capture_when_format_context_raises(self, monkeypatch):
        """The try/finally that makes this helper exist.

        If `format_context` raises, the capture handler MUST come
        off `kai.memory`'s logger. Otherwise the next probe in the
        process double-captures and aborts with "expected one log,
        got two." Assertion: handler count on the kai.memory logger
        is the same before and after the failing call.
        """

        async def fake_format_context(query: str, *, user_id: str, token_budget=None):
            raise ValueError("simulated retrieval failure")

        monkeypatch.setattr("kai.memory.format_context", fake_format_context)

        # Snapshot the logger's handler list before the call so the
        # assertion does not depend on whether pytest itself
        # attached any handlers; we only care that we did not LEAK
        # one.
        logger = logging.getLogger("kai.memory")
        handlers_before = list(logger.handlers)

        with pytest.raises(ValueError, match="simulated retrieval failure"):
            asyncio.run(legacy_retrieve_hits("any query", user_id="42"))

        handlers_after = list(logger.handlers)
        assert handlers_after == handlers_before, (
            "legacy_retrieve_hits leaked a capture handler on the kai.memory "
            "logger; the next probe in the process would double-capture"
        )
