"""
Unscoped recall-capture helpers.

Buffers the structured `memory.recall` log line that
`kai.memory.format_context` emits, parses it, and exposes
`legacy_retrieve_hits` as the public entry point for callers that
need the unscoped recall result for a single question rather than
the aggregate metrics the legacy eval CLI in `kai.eval.retrieval`
produces.

Why a dedicated module:

`kai.eval.retrieval` is deprecated (see its docstring). The
collision-probe generator in `kai.eval.gen_collision_probes` still
needs an unscoped verifier: when drafting a negative probe, the
question is "does this look like a hit under the OLD unscoped
gate?" and replacing the verifier with scoped retrieval would
reject every good negative. The helpers live here, under a name
that says what they are, so the eval CLI can be removed without
taking the generator's verifier path with it.

The capture machinery:

`format_context` emits a structured log line prefixed
`memory.recall ` (`_RECALL_PREFIX`) on every call. The capture
handler buffers matching records on the `kai.memory` logger; drain
returns the parsed JSON payloads. The handler is additive (the
existing log destinations still receive the lines) and forces the
logger's effective level to INFO at attach time so an operator
running with `LOG_LEVEL=WARNING` does not silently drop the records
the caller depends on.

The attach / detach pair is symmetric. `legacy_retrieve_hits` wraps
the full attach / format_context / drain / detach dance in a
try/finally so the capture handler is removed on every exit path,
including the case where `format_context` itself raises. A naive
caller that attaches and forgets to detach pollutes the logger for
the rest of the process and silently corrupts the next caller by
delivering its `memory.recall` record to two handlers.
"""

from __future__ import annotations

import json
import logging
from typing import Any

# Greppable prefix on every memory.recall log line. Mirrored in
# `kai.memory._emit_recall_log`; if either changes, the parser in
# `_RecallLogCapture` must move with it. The trailing space is part
# of the prefix so empty-payload edge cases still tokenize.
_RECALL_PREFIX = "memory.recall "


class _RecallLogCapture(logging.Handler):
    """Buffer memory.recall log records for later draining.

    Attached to the `kai.memory` logger so it sees every record the
    module emits. Filters down to records whose message starts with
    the `memory.recall ` prefix so unrelated info-level logs (init,
    delete, etc.) are ignored. The handler keeps full records, not
    just the parsed payloads, so per-record diagnostics (logger
    name, timestamp) remain accessible if a future debug hook needs
    them.

    Carries `_saved_level` so the attach helper can restore the
    `kai.memory` logger's effective level on detach. The attach
    helper forces the level to INFO because operators may run with
    LOG_LEVEL=WARNING in production - without the bump,
    `memory.recall` records would never reach any handler and any
    caller draining the buffer would see zero records and abort
    with "expected one log, got zero."
    """

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self._records: list[logging.LogRecord] = []
        self._saved_level: int | None = None

    def emit(self, record: logging.LogRecord) -> None:
        # `getMessage()` is the documented stable API; `.message` is
        # set as a side effect of Formatter.format() and is not
        # guaranteed to be populated for every handler.
        if record.getMessage().startswith(_RECALL_PREFIX):
            self._records.append(record)

    def drain(self) -> list[dict[str, Any]]:
        """Return parsed payloads for every buffered record and clear.

        Strips the `memory.recall ` prefix and decodes the remainder
        as JSON. Callers expect exactly one record per call to
        `format_context`; they assert on the count and abort loudly
        on zero or more than one - the log-shape contract is the
        only signal that the retrieval path ran as expected.
        """
        parsed: list[dict[str, Any]] = []
        for r in self._records:
            blob = r.getMessage()[len(_RECALL_PREFIX) :]
            try:
                payload = json.loads(blob)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"unparseable memory.recall payload: {e.msg}") from e
            parsed.append(payload)
        self._records.clear()
        return parsed


def _attach_capture() -> _RecallLogCapture:
    """Attach a recall-log capture handler to kai.memory's logger.

    Caller is responsible for detaching after use (see
    `_detach_capture`). The handler is additive (does NOT set
    propagate=False on the logger) so existing log destinations
    still receive the lines; the caller only intercepts a copy.

    Forces the `kai.memory` logger level to INFO if it is currently
    higher (e.g. WARNING in a quiet operator config). Without this,
    info-level `memory.recall` records would be filtered out before
    reaching any handler and the caller would abort with "expected
    one log, got zero." The original level is saved on the capture
    object so detach can restore it.
    """
    logger = logging.getLogger("kai.memory")
    capture = _RecallLogCapture()
    capture._saved_level = logger.level
    if logger.level == logging.NOTSET or logger.level > logging.INFO:
        logger.setLevel(logging.INFO)
    logger.addHandler(capture)
    return capture


def _detach_capture(capture: _RecallLogCapture) -> None:
    """Symmetric removal of the handler installed by _attach_capture.

    Restores the `kai.memory` logger's pre-attach level so a
    long-lived process running multiple capture sessions does not
    slowly accumulate verbosity changes.
    """
    logger = logging.getLogger("kai.memory")
    logger.removeHandler(capture)
    if capture._saved_level is not None:
        logger.setLevel(capture._saved_level)


async def legacy_retrieve_hits(question: str, user_id: str) -> tuple[list[dict[str, Any]], int]:
    """
    Run the legacy recall pipeline for one question and return its hits.

    Wraps the attach / format_context / drain / detach pattern so
    external callers (the collision-probe generator in
    `kai.eval.gen_collision_probes`) can reuse the legacy unscoped
    pipeline as a verification gate without reaching into
    `_attach_capture` / `_detach_capture` directly.

    Why a public helper and not just exposing the capture:
    callers need the attach / call / drain / detach invariant
    intact. Three of the four steps are bookkeeping around one
    awaited call that can raise; the only way to guarantee the
    capture handler is removed from `kai.memory`'s logger on every
    exit path is to keep the try/finally in one place. A naive
    caller that attaches and forgets to detach pollutes the logger
    for the rest of the process and silently corrupts the next
    caller by delivering its `memory.recall` record to two
    handlers.

    The single-payload check matches `format_context`'s contract
    (exactly one `memory.recall` line per call). Zero indicates the
    logger never received the record (level filter, missing
    handler, removed log statement); more than one indicates a
    logging-discipline regression. Both are raised loudly because
    silently picking one would make every downstream score wrong in
    ways nothing else would catch.

    Args:
        question: The probe question text. Passed verbatim to
            `kai.memory.format_context`.
        user_id: Kai user id (Telegram chat id as a string for
            production callers). Scopes the recall to one user.

    Returns:
        A 2-tuple of (hits, latency_ms) drawn from the captured
        `memory.recall` payload. `hits` is the raw list of hit
        dicts in the order the pipeline ranked them; `latency_ms`
        is the end-to-end retrieval latency the pipeline measured.

    Raises:
        RuntimeError: If zero or more than one `memory.recall` log
            records are captured for the call.
    """
    # Deferred import: `kai.memory` pulls in PyTorch /
    # sentence-transformers when memory is enabled, which is too
    # expensive to load at module-import time for callers that may
    # not actually invoke the helper.
    from kai.memory import format_context

    capture = _attach_capture()
    try:
        await format_context(question, user_id=user_id)
        payloads = capture.drain()
        if len(payloads) != 1:
            raise RuntimeError(
                f"expected exactly one memory.recall log per probe, got {len(payloads)} for question {question!r}"
            )
        payload = payloads[0]
        return list(payload.get("hits", [])), int(payload.get("latency_ms", 0))
    finally:
        # try/finally is the entire reason this helper exists. An
        # exception inside `format_context` (or inside the count
        # check) must NOT leave the capture handler attached to
        # `kai.memory`'s logger; the next caller in the process
        # would then double-capture and abort with "expected one
        # log, got two."
        _detach_capture(capture)
