"""
Provider-agnostic one-shot reasoning boundary.

Used by semantic memory extraction (and any future caller that needs a
bounded, stateless model call) so the caller does not embed provider
subprocess mechanics. Distinct from `kai.backend.AgentBackend`: that
abstraction is for persistent, interactive sessions that own a long-
running subprocess and inject Kai's identity / memory / history
context. A one-shot reasoner has different lifecycle requirements: no
persistent state, a hard per-call timeout, stdin-only prompt delivery,
optional structured-output schema, and an envelope that the CALLER
parses (the reasoner returns raw text, not parsed memory objects).

Phase 1 ships exactly one implementation: `ClaudeOneShotReasoner`. Its
argv, env allow-list, neutral cwd, and timeout semantics are lifted
verbatim from `kai.memory_extraction`'s previous direct subprocess
spawn so that the Claude memory path stays byte-identical to its
pre-refactor behavior. Future Codex / OpenCode implementations of the
same `OneShotReasoner` protocol live here as siblings.

`_EXTRACTOR_CWD`, `_ensure_extractor_cwd`, and `_SUBPROCESS_ENV_ALLOWLIST`
are canonical here, not in memory_extraction. The behavioral eval
harness (`kai.eval.behavioral`) reuses them to share the same neutral
cwd + env contract with memory extraction; both modules import from
this one. Re-exports in `memory_extraction.py` are one-line aliases
that point back here so test code that imports the old names
continues to resolve to the same objects.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from kai.config import DATA_DIR

log = logging.getLogger(__name__)


# ── Subprocess cwd / env ────────────────────────────────────────────


# Neutral cwd for the one-shot subprocess. The directory has no
# CLAUDE.md, no project files, and no checked-in conventions, so the
# spawned model cannot accidentally inherit Kai's workspace identity,
# voice, or instructions. `_ensure_extractor_cwd()` creates it lazily
# on first call; matches Kai's no-import-time-IO convention so a
# permissions or path failure surfaces as a logged extractor miss
# rather than an import-time crash that takes the bot down.
_EXTRACTOR_CWD = DATA_DIR / "memory" / "extractor_cwd"


# Env vars forwarded to the one-shot subprocess. The parent's full
# environment is deliberately NOT inherited: secrets like DATABASE_URL,
# GitHub tokens, webhook secrets, and Telegram tokens must not reach
# the model if the provider's tool-suppression flag ever regresses.
# The list below is the minimum needed for the Claude CLI to find its
# binary (PATH), read its config and OAuth state (HOME,
# CLAUDE_CONFIG_DIR), authenticate on the pay-per-token fallback
# (ANTHROPIC_API_KEY), and reach a proxy if one is configured
# (ANTHROPIC_BASE_URL). Vars absent from the parent env are simply
# not forwarded; the subprocess behaves as if they were unset.
_SUBPROCESS_ENV_ALLOWLIST = (
    "PATH",
    "HOME",
    "CLAUDE_CONFIG_DIR",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
)


def _ensure_extractor_cwd() -> None:
    """
    Create the neutral subprocess cwd on first use.

    Idempotent: mkdir(exist_ok=True) is cheap on the hot path. Called
    from `ClaudeOneShotReasoner.run()` before spawning the subprocess.
    Deferred from import time on purpose - a permission or path
    failure should surface as a logged miss, not an import-time crash
    that takes the whole bot down.
    """
    _EXTRACTOR_CWD.mkdir(parents=True, exist_ok=True)


# ── Protocol types ──────────────────────────────────────────────────


@dataclass(frozen=True)
class OneShotResult:
    """
    Return value from a `OneShotReasoner.run()` call.

    Attributes:
        text: The model's raw output text. For Claude in Phase 1 this
            is the stdout of `claude --print`, which the caller parses
            for the JSON envelope it expects (`is_error`,
            `structured_output`, etc.). The reasoner does not parse
            the envelope; that contract lives in memory_extraction.
        backend: Provider tag ("claude" today; "codex" / "opencode" in
            future implementations). Surfaces in structured logs and
            run-record JSON so cross-backend comparisons are possible.
        model: Model name passed to the provider, or None when the
            caller did not request a specific model. The reasoner
            does not validate model strings; that is the caller's
            responsibility.
        raw_metadata: Subprocess-layer metadata only - returncode,
            stderr bytes, and optionally cmd / cwd for log forensics.
            Phase 1 deliberately does NOT include parsed-envelope
            fields (is_error, total_cost_usd, etc.); those stay in
            memory_extraction so the envelope is parsed exactly once.
            Future provider implementations may populate
            backend-specific fields, but no Phase 1 caller depends on
            anything beyond returncode and stderr.
        duration_ms: Wall-clock duration of the underlying provider
            call, measured around the subprocess spawn + wait. Used
            for structured logging and for operator-side latency
            tracking; the value is informational, not a contract.
    """

    text: str
    backend: str
    model: str | None
    raw_metadata: dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0


class OneShotReasoner(Protocol):
    """
    Bounded, stateless model call.

    Implementations spawn a provider subprocess (or call a provider
    API), feed `prompt` as the user message, optionally prefix
    `system_prompt` and enforce the supplied JSON Schema, wait up to
    `timeout` seconds, and return a `OneShotResult`. Failures are
    typed exceptions (`OneShotTimeout`, `OneShotSubprocessError`,
    `OneShotOutputError`) the caller catches and maps to
    domain-specific failure semantics; the reasoner itself does not
    silently swallow errors or invent fallback values.

    `purpose` is a required string keyed to the caller's intent
    (`"fact_extraction"`, `"episode_generation"`, etc.). It appears
    in structured logs only; the reasoner does not branch on it. The
    field is required (not optional) so log streams can always
    identify which caller spawned the subprocess.
    """

    async def run(
        self,
        *,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        purpose: str,
        json_schema: dict[str, Any] | None = None,
    ) -> OneShotResult: ...


# ── Typed errors ────────────────────────────────────────────────────


class OneShotError(Exception):
    """Base for all reasoner-level failures. Memory callers catch this
    family and collapse it to domain-specific failure shapes; nothing
    in the reasoner itself raises bare Exception."""


class OneShotTimeout(OneShotError):
    """The provider subprocess exceeded the call's timeout and was
    killed. No payload fields in Phase 1: stage 1 maps this to
    `ExtractionResult(facts=[], has_episode=False)`, stage 2 to the
    literal reason `"timeout"`, neither of which needs additional
    context off the exception. If a future caller needs the elapsed
    time, add it then; do not anticipate."""


@dataclass
class OneShotSubprocessError(OneShotError):
    """
    The provider subprocess exited with a non-zero return code.

    Carries `returncode` and raw `stderr` bytes because stage 2's
    episode-extraction failure-reason string is
    `f"exit_{returncode}: {stderr[:200].decode('utf-8', errors='replace')}"`.
    Preserving that string is part of Acceptance 4 (stage 2 failure-
    reason shape). The fields are dataclass attributes, not
    constructor args, because Python's stdlib `Exception` does not
    play nicely with `__init__` parameter capture in subclasses;
    `@dataclass(eq=False)` on the subclass would also work but is not
    needed here because exception equality semantics are positional.
    """

    returncode: int
    stderr: bytes


class OneShotOutputError(OneShotError):
    """
    The provider returned successfully (rc=0, no timeout) but produced
    output the reasoner itself could not surface to the caller (e.g.,
    a stdout that is not valid UTF-8 even with errors='replace'). NOT
    raised for memory-domain failures such as schema mismatches,
    `is_error` envelopes, or missing fields; those are the caller's
    contract, not the reasoner's. Reserved for the cases where the
    reasoner cannot honestly produce a `OneShotResult`.
    """


# ── Claude implementation ───────────────────────────────────────────


class ClaudeOneShotReasoner:
    """
    Spawns `claude --print` once per call and returns the raw stdout.

    The argv, env allow-list, neutral cwd, and timeout semantics are
    lifted verbatim from the pre-refactor `_run_extractor` /
    `_run_episode_extractor` paths in `kai.memory_extraction`. That
    is the load-bearing invariant for the Phase 1 refactor: the
    Claude memory path must stay byte-identical to its prior
    behavior, so any future drift in this implementation is a
    semantic change and not a refactor.

    Flag rationale (carried over from the original site):

    - NO `--bare`. `--help` says it forces ANTHROPIC_API_KEY-only auth
      and bypasses OAuth / keychain, which would bypass Max-plan
      billing. The explicit flags below give equivalent sandboxing
      without that trade-off.
    - `--system-prompt` fully replaces the default; combined with a
      neutral cwd that has no CLAUDE.md, the spawned model cannot
      inherit Kai's workspace identity, voice, or operating rules.
    - `--tools ""` disables all built-in tools.
    - `--no-session-persistence` keeps `~/.claude/projects/` from
      growing a directory per call.
    - NO `--max-budget-usd`. On Max-plan OAuth the CLI's computed-
      cost ceiling has no relation to actual billing; runaway-loop
      protection comes from `asyncio.wait_for(timeout)` instead.
    - `--permission-mode bypassPermissions` is acceptable because
      `--tools ""` leaves nothing to permit or deny.

    Payload goes on stdin, not argv. Argv is visible via `ps -ef` and
    often captured by process accounting; stdin is not world-readable
    and survives a future multi-user transition without code changes.
    """

    def __init__(self, *, cwd: Path | None = None) -> None:
        """
        Args:
            cwd: Override for the subprocess working directory. Tests
                pass a temp path so the run does not touch the real
                DATA_DIR. Production callers leave this unset; the
                reasoner uses `_EXTRACTOR_CWD` and ensures it exists
                on first call.
        """
        self._cwd = cwd if cwd is not None else _EXTRACTOR_CWD

    async def run(
        self,
        *,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        purpose: str,
        json_schema: dict[str, Any] | None = None,
    ) -> OneShotResult:
        # Lazy mkdir: deferred from import time on purpose so a
        # permission failure here surfaces as a logged miss rather
        # than crashing the bot at startup.
        self._cwd.mkdir(parents=True, exist_ok=True)

        cmd: list[str] = ["claude", "--print"]
        if model is not None:
            cmd.extend(["--model", model])
        cmd.extend(["--output-format", "json"])
        if json_schema is not None:
            # `claude --print --json-schema <schema>` validates the
            # model's structured-output payload CLI-side. Stage 1 and
            # stage 2 both rely on this defense-in-depth gate; the
            # caller's Python validation is the primary contract, but
            # losing the CLI gate would let more malformed payloads
            # reach the parser. R2 in the spec calls this out.
            cmd.extend(["--json-schema", json.dumps(json_schema)])
        if system_prompt is not None:
            cmd.extend(["--system-prompt", system_prompt])
        cmd.extend(
            [
                "--permission-mode",
                "bypassPermissions",
                "--tools",
                "",
                "--no-session-persistence",
            ]
        )

        # Allow-listed env: only forward vars from
        # _SUBPROCESS_ENV_ALLOWLIST that are present in the parent
        # environment. Absent vars are simply not forwarded, matching
        # the prior parent-inherit semantics for "var was unset" but
        # without leaking the rest of the parent env. Defense-in-
        # depth against a future regression in `--tools ""`.
        subprocess_env: dict[str, str] = {
            key: os.environ[key] for key in _SUBPROCESS_ENV_ALLOWLIST if key in os.environ
        }

        start = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._cwd),
            env=subprocess_env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=prompt.encode("utf-8")),
                timeout=timeout,
            )
        except TimeoutError:
            # Match the pre-refactor cleanup: kill the subprocess and
            # await its reap before raising. Without the await, the
            # caller (e.g. _generate_episode) could see the
            # ProcessLookupError race on a subsequent kill.
            proc.kill()
            await proc.wait()
            duration_ms = int((time.monotonic() - start) * 1000)
            log.info(
                "oneshot_reasoner purpose=%s backend=claude model=%s duration_ms=%d outcome=timeout error_category=timeout",
                purpose,
                model,
                duration_ms,
            )
            raise OneShotTimeout() from None

        duration_ms = int((time.monotonic() - start) * 1000)
        returncode = proc.returncode if proc.returncode is not None else -1
        if returncode != 0:
            log.info(
                "oneshot_reasoner purpose=%s backend=claude model=%s duration_ms=%d outcome=subprocess_error error_category=non_zero_exit returncode=%d",
                purpose,
                model,
                duration_ms,
                returncode,
            )
            raise OneShotSubprocessError(returncode=returncode, stderr=stderr)

        log.info(
            "oneshot_reasoner purpose=%s backend=claude model=%s duration_ms=%d outcome=success returncode=0",
            purpose,
            model,
            duration_ms,
        )
        # Phase 1 raw_metadata: subprocess-layer data only. NO JSON
        # envelope parse here - that is memory_extraction's contract.
        # Future Codex / OpenCode implementations may populate
        # backend-specific metadata, but Phase 1 callers depend on
        # nothing beyond what is captured below.
        return OneShotResult(
            text=stdout.decode("utf-8", errors="replace"),
            backend="claude",
            model=model,
            raw_metadata={
                "returncode": returncode,
                "stderr": stderr,
                "cwd": str(self._cwd),
            },
            duration_ms=duration_ms,
        )
