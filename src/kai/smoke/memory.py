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

Invocation: `python -m kai.smoke.memory [--user-id <chat_id>] [--os-user <name>]`.

The `--user-id` flag selects which user's effective `default_backend`
the smoke runs under (the same per-user dispatch production uses).
When omitted, the smoke falls back to the global `config.default_backend`.
The `--os-user` flag is required when the resolved effective backend
is codex; the codex reasoner refuses to spawn without an os_user (it
exists to prevent the bot user from running codex). Pass the per-user
OS account name from `users.yaml`. The flag is accepted but ignored
on the claude reasoner branch; the historical Max-plan self-sudo-skip
path resolves None safely.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time

from kai.config import ONESHOT_REASONER_BACKENDS, ModelRole, get_model_for, load_config
from kai.memory_extraction import (
    _EXTRACTION_SYSTEM_PROMPT,
    _FACT_SCHEMA,
    _build_memory_reasoner,
    _resolve_effective_backend,
    _resolve_effective_provider,
)
from kai.oneshot import OneShotError

log = logging.getLogger("kai.smoke.memory")


# Fixed two-turn conversation. The user-asserted preference is stable
# enough that every backend's extractor reliably produces a single
# fact referencing the anchor substring; the assistant turn carries
# the conventional acknowledgement shape so the prompt's extraction
# rules fire on familiar ground.
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


async def _run(user_id: str | None, os_user: str | None) -> int:
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

    # Resolve the effective backend the smoke runs under. With per-user
    # dispatch (issue #515), production extraction uses each user's
    # effective `default_backend`; the smoke mirrors that by accepting a
    # `--user-id` that the helper looks up in users.yaml. Without
    # `--user-id`, fall back to the global `default_backend` (the legacy
    # single-backend smoke path). The string the helper takes is the
    # raw user_id (production threads it through `extract_and_store`
    # the same way), so reuse it directly.
    resolved_user_id = user_id if user_id is not None else "smoke"
    effective_backend = _resolve_effective_backend(resolved_user_id, config)
    effective_provider = _resolve_effective_provider(resolved_user_id, config)
    if effective_backend not in ONESHOT_REASONER_BACKENDS:
        eligible = ", ".join(sorted(ONESHOT_REASONER_BACKENDS))
        sys.stderr.write(
            f"smoke: effective backend {effective_backend!r} has no memory reasoner. "
            f"Pass --user-id matching an entry in users.yaml whose effective backend "
            f"is one of: {eligible}, or run with a matching global DEFAULT_BACKEND.\n"
        )
        return 1

    extraction_model = get_model_for(ModelRole.MEMORY_EXTRACTION, effective_backend, effective_provider)
    episode_model = get_model_for(ModelRole.MEMORY_EPISODE, effective_backend, effective_provider)
    print(f"backend: {effective_backend}")
    print(f"provider: {effective_provider}")
    print(f"extraction model: {extraction_model}")
    print(f"episode model: {episode_model}")
    print(f"user_id: {user_id or '(none; using global default_backend)'}")
    print(f"os_user: {os_user or '(none; direct spawn)'}")

    reasoner = _build_memory_reasoner(effective_backend, os_user=os_user)
    timeout = float(config.memory_extraction_timeout_s)

    start = time.monotonic()
    try:
        result = await reasoner.run(
            prompt=_build_payload(),
            system_prompt=_EXTRACTION_SYSTEM_PROMPT,
            model=extraction_model,
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
        "--user-id",
        dest="user_id",
        default=None,
        help=(
            "Telegram chat_id whose effective default_backend selects the reasoner. "
            "Defaults to the global default_backend when omitted."
        ),
    )
    parser.add_argument(
        "--os-user",
        dest="os_user",
        default=None,
        help=(
            "Per-user OS account for cross-user codex spawning. Required when the "
            "effective backend resolves to codex. Ignored on the claude branch."
        ),
    )
    args = parser.parse_args()

    # Configure logging at WARNING so the operator sees reasoner-side
    # failures inline without enabling Mem0 / Qdrant / urllib3 noise.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    return asyncio.run(_run(args.user_id, args.os_user))


if __name__ == "__main__":
    sys.exit(main())
