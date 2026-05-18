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
import contextlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from kai.codex_exec import extract_codex_text
from kai.config import DATA_DIR
from kai.prompt_utils import make_boundary

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


# ── Codex implementation ────────────────────────────────────────────


# Env vars forwarded to the codex one-shot subprocess. Deliberately
# separate from `_SUBPROCESS_ENV_ALLOWLIST`: that list carries
# Anthropic-specific variables (CLAUDE_CONFIG_DIR, ANTHROPIC_API_KEY,
# ANTHROPIC_BASE_URL) that mean nothing to codex and would only widen
# blast radius if forwarded. Codex authentication comes from one of
# two sources, both covered by this list:
#   - subscription (ChatGPT Pro) OAuth state under ~/.codex, reached
#     via HOME or the optional CODEX_HOME override.
#   - pay-per-token via OPENAI_API_KEY, with OPENAI_BASE_URL for
#     compatible proxies (Azure OpenAI, vLLM, etc.).
# PATH is required so the codex binary can resolve its own libexec
# helpers and any subprocesses it spawns. KAI_WEBHOOK_SECRET, GitHub
# tokens, Telegram tokens, DATABASE_URL, and the rest of the bot
# parent env are NOT forwarded; a regression that started reaching
# for them would surface as a logged miss rather than a silent
# secret-leak to the model.
_CODEX_ENV_ALLOWLIST = (
    "PATH",
    "HOME",
    "CODEX_HOME",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
)


def _render_codex_stdin(system_prompt: str | None, prompt: str) -> str:
    """
    Build a boundary-delimited stdin payload for a codex one-shot call.

    Codex `exec` mode has no `--system-prompt` flag. The closest
    equivalent is prepending the system text inside a randomized
    boundary block (same scheme `triage.py`, `review.py`, and the
    behavioral harness use to wrap untrusted prompt sections) and
    sending the user payload below it. The model treats the boundary
    block as authoritative context but cannot forge the closing token
    from inside the user payload because `make_boundary` re-rolls the
    token per call.

    A None or empty `system_prompt` produces no boundary block at all
    rather than an empty SYSTEM section. An empty SYSTEM section
    would emit a free-floating boundary marker that the model could
    see with no content inside it, which is both confusing and a
    pointless attack surface; sending the payload unchanged matches
    the no-system-prompt semantics callers already get from claude's
    `--system-prompt` flag being omitted.

    The helper lives in oneshot.py (not behavioral.py) deliberately:
    importing eval-harness code from a provider implementation would
    couple memory extraction's runtime to the eval module's import
    graph. The two helpers are intentional duplicates of a small
    function rather than a shared import, mirroring how `triage.py`
    and `review.py` each format their own boundary blocks.
    """
    if not system_prompt:
        return prompt
    begin, end = make_boundary("SYSTEM")
    return f"{begin}\n{system_prompt}\n{end}\n\n{prompt}"


class CodexOneShotReasoner:
    """
    Spawns `codex exec --json` once per call and returns either the
    raw final agent_message text or, when a JSON schema is supplied, a
    normalized `{"is_error": false, "structured_output": ...}` envelope
    string that matches the shape memory_extraction.py already parses.

    Codex differs from Claude in three load-bearing ways:

    - No `--system-prompt` flag: the system text is prepended to stdin
      via `_render_codex_stdin` with a randomized boundary block.
    - No inline `--json-schema`: codex takes `--output-schema <FILE>`,
      so this reasoner writes the schema to a per-call temp file and
      removes it on every exit path (success, timeout, non-zero exit,
      or output-parse failure).
    - NDJSON event output instead of a single envelope: stdout is a
      sequence of `thread.*`, `turn.*`, and `item.*` events. The final
      `agent_message` text is recovered via `extract_codex_text` with
      `join_items=False`, which matches triage's "last completed
      message only" contract for structured-output callers. A
      preamble agent_message followed by a JSON body would otherwise
      corrupt the parse.

    The schema-backed call always returns a wrapped envelope so the
    caller's nested-vs-root resolution path (the one already in
    `memory_extraction._run_extractor` and `_run_episode_extractor`)
    sees the same `structured_output` shape it sees from Claude. The
    reasoner does not inspect fact / episode / tag / confidence
    fields; that contract stays with the caller. When no schema is
    supplied (`json_schema is None`), the final agent_message text is
    returned directly without wrapping; memory extraction always
    supplies a schema, so the wrap path is the production memory
    path and the non-schema branch exists only for future free-form
    callers.

    Flag rationale:

    - `--json` is required for NDJSON event output; without it the
      CLI emits free-form text that the parser cannot walk.
    - `--skip-git-repo-check` is required because the neutral
      extractor cwd is not a Git repository. Codex's trusted-dir
      gate refuses to spawn otherwise; the lesson from the broader
      codex one-shot lineage is to pass this on every exec invocation.
    - `--ephemeral` mirrors Claude's `--no-session-persistence`: no
      session files written under ~/.codex per call.
    - `--ignore-rules` keeps user or project execpolicy `.rules`
      files from influencing memory extraction. The neutral cwd
      already avoids project rules, but the explicit flag protects
      the one-shot path from user-level rule drift on the service
      user's `~/.codex/`.
    - `--cd <cwd>` tells codex its working root explicitly; the
      subprocess `cwd` argument keeps the process environment aligned
      with that root so a future codex release that reads cwd from
      either source sees the same value.
    - `--output-schema <file>` is included only when a schema is
      supplied. Codex's schema enforcement is best-effort (unlike
      claude's CLI-side `--json-schema` validation); the reasoner
      parses the final text as JSON and raises `OneShotOutputError`
      on malformed output so memory storage never sees a half-shaped
      payload.

    Deliberately NOT passed:

    - `--dangerously-bypass-approvals-and-sandbox`: memory extraction
      asks for structured output only and does not need tools, shell
      commands, or filesystem writes. Granting the bypass would widen
      blast radius if a future codex release tried to invoke tools.
    - `--sandbox danger-full-access`: same reasoning. If codex
      attempts tool calls during exec, the call should fail or
      produce no valid final JSON; the reasoner then raises
      `OneShotOutputError` and the caller stores nothing.

    The schema temp file lives under `DATA_DIR/memory/extractor_cwd/`
    on production and the test-supplied cwd otherwise. Storing it
    next to the run cwd keeps the path predictable in logs and means
    a crash that leaks the file leaves it inside the memory subtree
    rather than under /tmp where unrelated tooling might trip on it.
    Cleanup is wrapped in `contextlib.suppress(FileNotFoundError)` so
    a process death between write and remove does not leave a stale
    exception masking the original failure.
    """

    def __init__(self, *, cwd: Path | None = None) -> None:
        """
        Args:
            cwd: Override for the subprocess working directory and the
                schema temp file's parent. Tests pass a temp path so
                the run does not touch the real DATA_DIR. Production
                callers leave this unset; the reasoner uses
                `_EXTRACTOR_CWD` and ensures it exists on first call.
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
        # Same lazy-mkdir contract as Claude: deferred from import
        # time so a permission failure surfaces as a logged miss
        # rather than crashing the bot at startup.
        self._cwd.mkdir(parents=True, exist_ok=True)

        codex_bin = os.environ.get("CODEX_BIN") or "codex"
        cmd: list[str] = [
            codex_bin,
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--ephemeral",
            "--ignore-rules",
            "--cd",
            str(self._cwd),
        ]
        if model is not None:
            cmd.extend(["--model", model])

        # Schema temp file lifecycle. Written before subprocess spawn
        # so the CLI sees a populated file; removed in `finally` so
        # success, timeout, non-zero exit, and output-parse failures
        # all clean up. `delete=False` is required because the
        # subprocess opens the file by path, not by inherited fd;
        # NamedTemporaryFile's auto-delete on close would yank the
        # file out from under the running codex process.
        schema_path: Path | None = None
        if json_schema is not None:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                prefix="codex-schema-",
                dir=str(self._cwd),
                delete=False,
                encoding="utf-8",
            ) as fh:
                json.dump(json_schema, fh)
                schema_path = Path(fh.name)
            cmd.extend(["--output-schema", str(schema_path)])

        # Allow-listed env: only forward keys present in the parent
        # env. Absent keys are simply not forwarded; the subprocess
        # behaves as if they were unset. Defense-in-depth against a
        # future regression that tries to reuse Claude's env list:
        # the codex list deliberately excludes the Anthropic-specific
        # variables and keeps the secret surface narrow.
        subprocess_env: dict[str, str] = {key: os.environ[key] for key in _CODEX_ENV_ALLOWLIST if key in os.environ}

        stdin_payload = _render_codex_stdin(system_prompt, prompt)

        start = time.monotonic()
        try:
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
                    proc.communicate(input=stdin_payload.encode("utf-8")),
                    timeout=timeout,
                )
            except TimeoutError:
                # Same kill+await pattern as Claude: without the await
                # a subsequent kill on the reaped pid could race on
                # ProcessLookupError. The duration is measured around
                # the wait_for so the log line shows wall-clock
                # cost-to-timeout, not wall-clock-to-reap.
                proc.kill()
                await proc.wait()
                duration_ms = int((time.monotonic() - start) * 1000)
                log.info(
                    "oneshot_reasoner purpose=%s backend=codex model=%s duration_ms=%d outcome=timeout error_category=timeout",
                    purpose,
                    model,
                    duration_ms,
                )
                raise OneShotTimeout() from None

            duration_ms = int((time.monotonic() - start) * 1000)
            returncode = proc.returncode if proc.returncode is not None else -1
            if returncode != 0:
                log.info(
                    "oneshot_reasoner purpose=%s backend=codex model=%s duration_ms=%d outcome=subprocess_error error_category=non_zero_exit returncode=%d",
                    purpose,
                    model,
                    duration_ms,
                    returncode,
                )
                raise OneShotSubprocessError(returncode=returncode, stderr=stderr)

            stdout_text = stdout.decode("utf-8", errors="replace")
            # Triage's lesson carried over: structured-output callers
            # want only the LAST completed agent_message so a preamble
            # message does not get glued ahead of the JSON body. Memory
            # extraction is the canonical structured caller here; the
            # `join_items=True` (review/chat) path has no caller in
            # #497 but the protocol leaves room for one.
            final_text = extract_codex_text(stdout_text, join_items=False)

            if json_schema is None:
                # Free-form path: hand the final agent_message text
                # back unchanged. Memory extraction never takes this
                # branch in production; it stays here so a future
                # free-form caller does not need to special-case
                # codex.
                log.info(
                    "oneshot_reasoner purpose=%s backend=codex model=%s duration_ms=%d outcome=success returncode=0",
                    purpose,
                    model,
                    duration_ms,
                )
                return OneShotResult(
                    text=final_text,
                    backend="codex",
                    model=model,
                    raw_metadata={
                        "returncode": returncode,
                        "stderr": stderr,
                        "cwd": str(self._cwd),
                    },
                    duration_ms=duration_ms,
                )

            # Schema-backed path. The final agent_message must parse
            # as a JSON object; anything else is OneShotOutputError so
            # memory_extraction's caller-side mapping collapses it to
            # the zero-state extraction result instead of letting a
            # malformed payload reach the fact validator.
            if not final_text:
                log.info(
                    "oneshot_reasoner purpose=%s backend=codex model=%s duration_ms=%d outcome=output_error error_category=empty_agent_message returncode=0",
                    purpose,
                    model,
                    duration_ms,
                )
                raise OneShotOutputError("codex produced no final agent_message")
            try:
                payload = json.loads(final_text)
            except json.JSONDecodeError as exc:
                log.info(
                    "oneshot_reasoner purpose=%s backend=codex model=%s duration_ms=%d outcome=output_error error_category=invalid_json returncode=0",
                    purpose,
                    model,
                    duration_ms,
                )
                raise OneShotOutputError(f"codex final text was not valid JSON: {exc}") from None

            # Codex's `--output-schema` enforcement is best-effort:
            # the CLI does not hard-reject a final message that
            # parses as JSON but does not match the schema. Three
            # distinct shape failures must surface as typed errors,
            # not as a successful is_error=false envelope that the
            # caller would treat as "the model found nothing":
            #
            #   1. Scalar / list / null payloads (`json.loads` returns
            #      something that is not a dict).
            #   2. Object payloads missing required top-level fields
            #      named by the supplied schema (e.g. an `{"episode":
            #      ...}` response under the fact schema, which
            #      requires `facts` and `has_episode`).
            #
            # The required-field check intentionally stays minimal:
            # the reasoner does not import a JSON Schema validator
            # because the runtime dependency is not worth the cost
            # for two callers, and a deeper structural check belongs
            # to the caller's own `_validate_facts` / episode
            # validators. Only the top-level required list is read
            # off `json_schema`; anything beyond that (additional
            # properties, nested required, type enforcement) is left
            # to the caller's existing validators, same posture as
            # the Claude path.
            if not isinstance(payload, dict):
                log.info(
                    "oneshot_reasoner purpose=%s backend=codex model=%s duration_ms=%d outcome=output_error error_category=non_object_json returncode=0",
                    purpose,
                    model,
                    duration_ms,
                )
                raise OneShotOutputError("codex final JSON was not an object")

            required_fields = json_schema.get("required")
            if isinstance(required_fields, list):
                missing = [field for field in required_fields if field not in payload]
                if missing:
                    log.info(
                        "oneshot_reasoner purpose=%s backend=codex model=%s duration_ms=%d outcome=output_error error_category=missing_required_fields returncode=0",
                        purpose,
                        model,
                        duration_ms,
                    )
                    raise OneShotOutputError(f"codex final JSON missing required fields: {missing}")

            # Wrap codex's schema-shaped payload in the same envelope
            # claude emits natively. memory_extraction.py already
            # walks the `structured_output` field on a parsed dict;
            # rewrapping here keeps the parser path provider-neutral.
            # `is_error: false` is hardcoded because reaching this
            # branch means rc=0 AND a parseable final JSON; the
            # caller's is_error guard still fires on a future codex
            # variant that emits an error envelope as its agent_message.
            envelope = json.dumps({"is_error": False, "structured_output": payload})
            log.info(
                "oneshot_reasoner purpose=%s backend=codex model=%s duration_ms=%d outcome=success returncode=0",
                purpose,
                model,
                duration_ms,
            )
            return OneShotResult(
                text=envelope,
                backend="codex",
                model=model,
                raw_metadata={
                    "returncode": returncode,
                    "stderr": stderr,
                    "cwd": str(self._cwd),
                },
                duration_ms=duration_ms,
            )
        finally:
            # Cleanup runs on every exit path: success, timeout,
            # subprocess error, output error, and any unexpected
            # exception propagating from the spawn itself. Suppress
            # FileNotFoundError so a cleanup race (e.g. the OS
            # already removed the file under a temp sweep) does not
            # mask the original failure with a secondary one.
            if schema_path is not None:
                with contextlib.suppress(FileNotFoundError):
                    schema_path.unlink()
