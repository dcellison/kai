#!/usr/bin/env python3
"""
Episode-classifier probe (issue #392).

Spawns `claude --print` with the production stage-1 extractor flags and
drives a small labeled corpus through the windowed payload builder.
For each labeled case prints per-case `has_episode`, `facts_count`,
and the input payload size, plus a final pass/fail summary against
the expected `has_episode` labels.

This is the manual-verification gate referenced by issue #392's
acceptance criterion: a post-merge run reproduces the (true, false)
discrimination on the labeled corpus, which is what the spec was
designed against. Failure to reproduce post-merge means the production
wiring diverged from the design and needs investigation.

Usage:
    python scripts/episode-classifier-probe.py [--turns N] [--model NAME]

`--turns` overrides the windowed-payload `prior_pairs` slice size
(default 3, matching the dataclass default in `Config`). `--model`
overrides the extractor model (default `claude-haiku-4-5-20251001`,
matching `Config.memory_extraction_model`'s dataclass default).

Imports `_FACT_SCHEMA`, `_EXTRACTION_SYSTEM_PROMPT`,
`_SUBPROCESS_ENV_ALLOWLIST`, and `_build_extraction_payload` directly
from `kai.memory_extraction` so the probe uses the same prompt + schema
+ env-allowlist + payload format the production extractor uses. Drift
between probe and production should fail loudly here rather than
silently masking real classifier behavior.

The run is unbilled under the subscription-auth deployment posture;
operators can iterate on the corpus and prompt freely.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Make src/ importable so we re-use the production prompt + schema +
# env allowlist + payload builder rather than inlining copies that
# would silently drift. Same pattern as scripts/measure-extraction-timing.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kai.memory_extraction import (
    _EXTRACTION_SYSTEM_PROMPT,
    _FACT_SCHEMA,
    _SUBPROCESS_ENV_ALLOWLIST,
    _build_extraction_payload,
)

# Default model matches `Config.memory_extraction_model` dataclass
# default. Hardcoded rather than loaded via kai.config.load_config() so
# the script does not require a working production env (TELEGRAM_BOT_TOKEN
# etc.) just to run a labeled probe.
_DEFAULT_MODEL = "claude-haiku-4-5-20251001"


@dataclass(frozen=True)
class _LabeledCase:
    """One labeled-corpus entry. `prior_pairs` simulates the windowed
    PRIOR CONTEXT block (`bot.py` would normally fetch these from
    JSONL via `history.get_recent_pairs`); the (user, assistant) pair
    is the CURRENT EXCHANGE under judgment. `expected_has_episode` is
    the human label the probe asserts against."""

    name: str
    prior_pairs: list[tuple[str, str]]
    user: str
    assistant: str
    expected_has_episode: bool


# Labeled corpus. Inlined rather than read from a fixture file so the
# script is a single-file commit. Operators iterating on prompt
# revisions can edit this directly. The two required cases per issue
# #392 acceptance: one episode-shaped exchange where closure is in
# the current turn, and one analytical-question exchange where the
# current turn requests information without resolving.
_LABELED_CORPUS: list[_LabeledCase] = [
    _LabeledCase(
        name="decision-arc-with-closure",
        # Three prior turns establish a deliberation about which
        # provider to pick; the current turn closes the loop with a
        # stated decision. Classifier should return has_episode=true.
        prior_pairs=[
            (
                "I'm trying to pick a provider for embedding generation. "
                "What are the tradeoffs between Anthropic-side embeddings "
                "and a local model like all-MiniLM-L6-v2?",
                "Anthropic does not currently expose a public embeddings "
                "API. Your real options are OpenAI's text-embedding-3 "
                "family (cloud, paid) or local sentence-transformers "
                "models like all-MiniLM-L6-v2 (free, ~100MB, runs on CPU). "
                "For a local-deployment-first project, MiniLM is the "
                "default fit.",
            ),
            (
                "What about embedding quality? Will MiniLM be good enough for memory retrieval?",
                "MiniLM produces 384-dim embeddings that work well for "
                "short-to-medium semantic similarity tasks. For very "
                "long documents or cross-lingual retrieval, you would "
                "want a larger model, but for personal-assistant memory "
                "retrieval (short fact strings, English-only) it is more "
                "than adequate. Latency is ~5ms per encode on Apple "
                "Silicon.",
            ),
            (
                "OK and what about size on disk? I want to keep the footprint small.",
                "MiniLM is ~90MB on disk for the quantized variant or "
                "~100MB unquantized. Negligible compared to the rest of "
                "the project's dependency footprint.",
            ),
        ],
        user=(
            "Alright, I'm going with MiniLM. Local-first beats the complexity of an external API call for this project."
        ),
        assistant=(
            "Sounds good. I'll wire MiniLM as the default in the memory "
            "config; you can override via env var if you want to switch "
            "later."
        ),
        expected_has_episode=True,
    ),
    _LabeledCase(
        name="analytical-question-no-closure",
        # Empty prior context (a brand-new conversation) plus an
        # analytical question with no decision in the current turn.
        # Classifier should return has_episode=false.
        prior_pairs=[],
        user="What's the difference between async and threading in Python?",
        assistant=(
            "Both let you run multiple things concurrently, but they "
            "differ in mechanism. Threading uses OS-level threads "
            "scheduled preemptively by the OS; async uses cooperative "
            "scheduling within a single thread, where coroutines yield "
            "control at await points. Threading is the right pick for "
            "CPU-bound work that releases the GIL (numpy, I/O); async "
            "shines for high-concurrency I/O-bound work like HTTP "
            "servers handling thousands of connections. Pick async for "
            "network code, threading for parallel compute that uses "
            "C extensions."
        ),
        expected_has_episode=False,
    ),
]


async def _run_one(case: _LabeledCase, model: str, prior_turns: int, timeout_s: int) -> dict:
    """
    Spawn one extraction subprocess for the labeled case and return a
    per-case result dict. The dict carries the parsed extractor
    output (`has_episode`, `facts` list) plus payload metadata.

    Mirrors the production stage-1 invocation in
    `kai.memory_extraction._run_extractor` (no cost cap on argv;
    allow-listed env per the original sandboxing rationale;
    `asyncio.wait_for` around `proc.communicate` so a hung Haiku call
    surfaces as an explicit error rather than freezing the operator's
    terminal). The probe diverges from production only by reading the
    labeled `prior_pairs` directly from the corpus rather than via
    `history.get_recent_pairs`. The goal is to exercise the classifier
    prompt with controlled inputs, not to also test the JSONL
    retrieval path.
    """
    # Slice the labeled prior pairs to the configured window. The
    # default (3) matches the production default; operators tuning N
    # for false-positive analysis can pass --turns to widen or
    # narrow the window without editing the corpus.
    prior_pairs = case.prior_pairs[-prior_turns:] if prior_turns > 0 else []

    cmd = [
        "claude",
        "--print",
        "--model",
        model,
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(_FACT_SCHEMA),
        "--system-prompt",
        _EXTRACTION_SYSTEM_PROMPT,
        "--permission-mode",
        "bypassPermissions",
        "--tools",
        "",
        "--no-session-persistence",
    ]
    payload = _build_extraction_payload(
        case.user,
        case.assistant,
        prior_pairs=prior_pairs,
    )
    payload_bytes = payload.encode("utf-8")
    # Build the allow-listed env to match production's defense-in-depth
    # against a regression in `--tools ""`. Same pattern as
    # `_run_extractor` / `_run_episode_extractor` in memory_extraction.py.
    subprocess_env: dict[str, str] = {key: os.environ[key] for key in _SUBPROCESS_ENV_ALLOWLIST if key in os.environ}

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=subprocess_env,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=payload_bytes),
            timeout=timeout_s,
        )
    except TimeoutError:
        # Hung Haiku call. Reap the subprocess so operators running
        # the probe interactively are not left with an orphan claude
        # process draining battery. Return an error result so the
        # final summary surfaces the timeout case as a failure.
        proc.kill()
        await proc.wait()
        return {
            "case": case.name,
            "error": f"subprocess timed out after {timeout_s}s",
            "payload_size": len(payload_bytes),
        }

    if proc.returncode != 0:
        return {
            "case": case.name,
            "error": f"subprocess exit {proc.returncode}: {stderr.decode('utf-8', errors='replace')[:500]}",
            "payload_size": len(payload_bytes),
        }

    # The CLI envelope wraps the model's structured output. The
    # extractor's JSON object lives at envelope["result"] as a JSON
    # STRING (per claude --print's structured-output convention), so
    # we json.loads twice: once for the envelope, once for the inner
    # payload.
    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError:
        return {
            "case": case.name,
            "error": "envelope JSON decode failed",
            "payload_size": len(payload_bytes),
            "stdout": stdout.decode("utf-8", errors="replace")[:500],
        }
    inner_raw = envelope.get("result", "")
    try:
        parsed = json.loads(inner_raw)
    except json.JSONDecodeError:
        return {
            "case": case.name,
            "error": "inner JSON decode failed",
            "payload_size": len(payload_bytes),
            "result": inner_raw[:500],
        }

    return {
        "case": case.name,
        "has_episode": parsed.get("has_episode"),
        "facts_count": len(parsed.get("facts", [])),
        "payload_size": len(payload_bytes),
        "expected_has_episode": case.expected_has_episode,
    }


async def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--turns",
        type=int,
        default=3,
        help="Number of prior exchanges to include in the windowed payload (default: 3).",
    )
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=f"Extractor model id (default: {_DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=90,
        help=(
            "Per-case subprocess timeout in seconds (default: 90). Matches the "
            "production-tuned MEMORY_EXTRACTION_TIMEOUT_S; raise it if Haiku is "
            "running slowly under load."
        ),
    )
    args = parser.parse_args()

    print(f"Episode-classifier probe (turns={args.turns}, model={args.model}, timeout={args.timeout}s)")
    print(f"Cases: {len(_LABELED_CORPUS)}")
    print()

    results = []
    for case in _LABELED_CORPUS:
        result = await _run_one(case, args.model, args.turns, args.timeout)
        results.append(result)
        # Pretty-print one case at a time so an operator watching live
        # output can see progress on a slow Haiku call.
        print(f"[{case.name}]")
        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue
        actual = result["has_episode"]
        expected = result["expected_has_episode"]
        status = "PASS" if actual == expected else "FAIL"
        print(f"  has_episode={actual}  expected={expected}  [{status}]")
        print(f"  facts_count={result['facts_count']}")
        print(f"  payload_size={result['payload_size']} bytes")
        print()

    # Summary: count pass/fail. A run with any FAIL is a signal that
    # production diverged from the spec or that the prompt regressed
    # under live data; either way it warrants investigation.
    passes = sum(1 for r in results if "error" not in r and r["has_episode"] == r["expected_has_episode"])
    fails = sum(1 for r in results if "error" not in r and r["has_episode"] != r["expected_has_episode"])
    errors = sum(1 for r in results if "error" in r)
    print(f"Summary: {passes} pass, {fails} fail, {errors} error (out of {len(results)})")
    # Exit non-zero on any fail or error so a CI-style invocation can
    # surface a regression. A clean run is exit 0.
    return 0 if fails == 0 and errors == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
