"""End-to-end smoke for the memory reasoner pipeline.

Runs ONE extraction call against a fixed two-turn conversation and
prints the resolved backend, model, invoked binary, extracted facts,
episode-classification result, and total latency. Exit 0 when the
extraction returns at least one fact whose content references the
expected anchor; non-zero otherwise.

Does NOT touch the Qdrant store. The reasoner runs against the
in-memory fact schema and the result is parsed and printed; nothing
is persisted. Safe to invoke repeatedly without polluting the
install's memory state.

Invocation: `python -m kai.smoke.memory [--os-user <name>]`.

The `--os-user` flag is required when the configured reasoner is
codex; the codex reasoner refuses to spawn without an os_user (it
exists to prevent the bot user from running codex). Pass the per-
user OS account name from `users.yaml`. The flag is accepted but
ignored on the claude reasoner branch; the historical Max-plan
self-sudo-skip path resolves None safely.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time

from kai.config import load_config
from kai.memory_extraction import (
    _EXTRACTION_SYSTEM_PROMPT,
    _FACT_SCHEMA,
    _get_memory_reasoner,
)
from kai.oneshot import OneShotError

log = logging.getLogger("kai.smoke.memory")


# Fixed two-turn conversation. The user-asserted preference is stable
# enough that both claude and codex extractors reliably produce a
# single fact referencing the anchor substring; the assistant turn
# carries the conventional acknowledgement shape so the prompt's
# extraction rules fire on familiar ground.
_SMOKE_USER_TEXT = "Quick note: I take my coffee with oat milk, no sugar."
_SMOKE_ASSISTANT_TEXT = "Noted, coffee with oat milk and no sugar."
_SMOKE_ANCHOR_SUBSTRING = "oat milk"


def _build_payload() -> str:
    """Render the smoke conversation as the extractor's expected
    prompt shape. Same pattern as `_build_extraction_payload`'s output
    but inline so the smoke does not couple to its internal builders;
    a future change to the builder should be re-derived rather than
    silently picked up here."""
    return f"[CURRENT EXCHANGE]\nUser: {_SMOKE_USER_TEXT}\nAssistant: {_SMOKE_ASSISTANT_TEXT}\n"


async def _run(os_user: str | None) -> int:
    """Execute the smoke and return the desired exit code.

    Returns 0 on success (anchor substring appears in at least one
    extracted fact), 1 on any failure path (reasoner raises, JSON
    envelope unparseable, no facts, no anchor hit). Exception
    messages are logged at WARNING; the operator sees them in the
    terminal output."""
    config = load_config()

    # Production-mode precondition. The smoke claims to verify the
    # memory extraction pipeline end-to-end; running the reasoner
    # against a fixed payload when production has memory disabled or
    # extraction disabled produces a false-positive "pass" for a path
    # that will never fire on real traffic. Refuse to run in those
    # configurations rather than print a misleading verdict.
    if not config.memory_enabled:
        sys.stderr.write(
            "smoke: MEMORY_ENABLED is false; memory extraction is not configured to run "
            "in production. The smoke would exercise the reasoner against a fixed payload "
            "but that result does not reflect production behavior. Enable memory in the "
            "wizard before running the smoke.\n"
        )
        return 1
    if not config.memory_extraction_enabled:
        sys.stderr.write(
            "smoke: MEMORY_EXTRACTION_ENABLED is false (retrieval-only mode); memory "
            "extraction is not configured to run in production. The smoke verifies the "
            "extraction pipeline; for retrieval-only installs, exercise retrieval via "
            "the /memory commands instead. Enable extraction in the wizard to run the "
            "smoke.\n"
        )
        return 1

    if config.memory_reasoner_backend == "codex" and not os_user:
        sys.stderr.write(
            "smoke: --os-user is required when memory_reasoner_backend=codex; "
            "pass the per-user OS account configured in users.yaml.\n"
        )
        return 1

    print(f"backend: {config.memory_reasoner_backend}")
    print(f"extraction model: {config.memory_extraction_model}")
    print(f"episode model: {config.memory_episode_model or '(inherits extraction model)'}")
    print(f"os_user: {os_user or '(none; direct spawn)'}")

    reasoner = _get_memory_reasoner(config, os_user=os_user)
    timeout = float(config.memory_extraction_timeout_s)

    start = time.monotonic()
    try:
        result = await reasoner.run(
            prompt=_build_payload(),
            system_prompt=_EXTRACTION_SYSTEM_PROMPT,
            model=config.memory_extraction_model,
            timeout=timeout,
            purpose="smoke_memory",
            json_schema=_FACT_SCHEMA,
        )
    except OneShotError as e:
        latency_ms = int((time.monotonic() - start) * 1000)
        log.warning("reasoner failed after %d ms: %s", latency_ms, e)
        print(f"latency_ms: {latency_ms}")
        print(f"error: {type(e).__name__}: {e}")
        return 1

    latency_ms = int((time.monotonic() - start) * 1000)

    # The reasoner records the actually invoked argv plus the
    # pre-sudo agent binary path in raw_metadata. Print both: cmd
    # for forensics under cross-user wrapping (where cmd[0] is
    # "sudo"), resolved_binary for the "which binary actually ran"
    # answer the operator usually wants.
    resolved_binary = result.raw_metadata.get("resolved_binary") or "(not recorded)"
    cmd = result.raw_metadata.get("cmd") or []
    print(f"resolved_binary: {resolved_binary}")
    print(f"argv: {' '.join(cmd) if cmd else '(not recorded)'}")
    print(f"latency_ms: {latency_ms}")

    # Parse the envelope the reasoner returned. Claude returns
    # `--output-format=json` shape; codex returns the
    # `{is_error, structured_output}` rewrap.
    try:
        envelope = json.loads(result.text)
    except json.JSONDecodeError as e:
        log.warning("could not parse reasoner output as JSON: %s", e)
        print(f"raw_text (first 300 chars): {result.text[:300]!r}")
        return 1

    structured = envelope.get("structured_output")
    if not isinstance(structured, dict):
        log.warning("envelope missing structured_output: keys=%s", sorted(envelope))
        return 1

    facts = structured.get("facts") or []
    has_episode = structured.get("has_episode")
    print(f"has_episode: {has_episode}")
    print(f"fact_count: {len(facts)}")
    for i, fact in enumerate(facts):
        content = fact.get("content", "")
        tags = fact.get("tags", [])
        speaker = fact.get("speaker", "")
        intent = fact.get("intent", "")
        print(f"  fact[{i}]: speaker={speaker} intent={intent} tags={tags} content={content!r}")

    # Success contract: at least one fact whose content references
    # the anchor substring. The smoke does NOT enforce a specific
    # extraction shape beyond that; both backends sometimes
    # paraphrase the user assertion.
    anchor_hit = any(_SMOKE_ANCHOR_SUBSTRING.lower() in (f.get("content") or "").lower() for f in facts)
    if not anchor_hit:
        log.warning(
            "no fact referenced the anchor substring %r; extraction succeeded but content did not match.",
            _SMOKE_ANCHOR_SUBSTRING,
        )
        return 1
    print("verdict: pass")
    return 0


def main() -> int:
    """Entry point for `python -m kai.smoke.memory`."""
    parser = argparse.ArgumentParser(
        prog="python -m kai.smoke.memory",
        description="Verify the memory reasoner pipeline against a fixed payload.",
    )
    parser.add_argument(
        "--os-user",
        dest="os_user",
        default=None,
        help=(
            "Per-user OS account for cross-user codex spawning. Required when "
            "MEMORY_REASONER_BACKEND=codex. Ignored on the claude reasoner branch."
        ),
    )
    args = parser.parse_args()

    # Configure logging at WARNING so the operator sees reasoner-side
    # failures inline without enabling Mem0 / Qdrant / urllib3 noise.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    return asyncio.run(_run(args.os_user))


if __name__ == "__main__":
    sys.exit(main())
