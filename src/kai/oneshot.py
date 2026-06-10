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
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from kai.acp import drain_late_text, read_result, write_rpc
from kai.codex_exec import extract_codex_text
from kai.config import DATA_DIR, resolve_claude_user
from kai.oneshot_binary import BinaryResolutionError, resolve_oneshot_binary
from kai.opencode import (
    concat_opencode_text,
    extract_opencode_text_delta,
    handle_opencode_permission_request,
)
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
    Create the neutral subprocess cwd on first use and chmod it 0o755
    on every call.

    Idempotent on the create side: `mkdir(exist_ok=True)` is cheap on
    the hot path. The unconditional `chmod(0o755)` is the load-bearing
    line for cross-user routing: when memory extraction routes to a
    different `os_user` via `sudo`, that user must be able to enter
    the cwd as their working directory. A pre-existing tighter mode
    (e.g., `0o700` from a stricter site umask, or the historical
    default for `mkdir` before this spec) would silently break the
    routing as a `subprocess_error` with `Permission denied` in
    stderr. Running the chmod every call costs one stat syscall and
    self-heals any pre-existing tighter mode without requiring the
    operator to reset state.

    Deferred from import time on purpose - a permission or path
    failure should surface as a logged miss, not an import-time crash
    that takes the whole bot down.
    """
    _EXTRACTOR_CWD.mkdir(parents=True, exist_ok=True)
    _EXTRACTOR_CWD.chmod(0o755)


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


class OneShotRoutingError(OneShotError):
    """
    Reasoner refused to run for a routing reason that cannot be
    satisfied at the call site. The only current source is binary
    resolution failure: `resolve_oneshot_binary` raised
    `BinaryResolutionError` and the reasoner re-raises as this
    typed error so the caller's existing `OneShotError` catch
    surface collapses to the zero-state extraction result. Both
    backends now accept `os_user=None` (the self-sudo-skip path),
    so missing-os_user is no longer a refusal source.
    """


# ── Per-backend env-preservation helpers ────────────────────────────


# Auth variables that must survive `sudo --preserve-env=...` so the
# wrapped agent can authenticate. HOME is deliberately omitted because
# `-H` rewrites it to the target user's pw entry; PATH is omitted
# because the bot user's PATH is forwarded into sudo's environment via
# the subprocess env allow-list (and sudo resolves bare commands like
# `claude` / `codex` against that PATH before the sudoers rule
# matches the resolved absolute path).
_CLAUDE_PRESERVED_AUTH_VARS: tuple[str, ...] = (
    "CLAUDE_CONFIG_DIR",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
)
_CODEX_PRESERVED_AUTH_VARS: tuple[str, ...] = (
    "CODEX_HOME",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
)
# OpenCode authenticates via `opencode auth login` which writes to
# `~/.local/share/opencode/auth.json` under the target user's HOME.
# `sudo -H` rewrites HOME to the target user, so the credentials file
# resolves automatically; what must survive `env_reset` are the per-
# provider API keys operators commonly export when their auth flow
# does not use the on-disk auth.json (e.g., a CI secret, a per-shell
# `direnv` injection, or a temporary key from a vault). HOME and PATH
# are NOT listed here for the same reason they are omitted from the
# claude / codex lists: PATH comes through the subprocess env allow-
# list and HOME is rewritten by `sudo -H`.
_OPENCODE_PRESERVED_AUTH_VARS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
    "DEEPSEEK_API_KEY",
)


def _preserved_auth_vars_for(backend: str) -> tuple[str, ...]:
    """
    Per-backend list of env vars that must survive sudo's `env_reset`.

    Threaded into the `--preserve-env=<csv>` flag on the wrap path so
    auth state the agent needs to reach reaches the target subprocess.
    Sudoers entries generated by `install._generate_sudoers` carry the
    `SETENV:` tag, which permits these vars to pass through. Sourced
    here so a future allow-list change in the subprocess env happens
    alongside the preserve list rather than drifting silently.
    """
    if backend == "claude":
        return _CLAUDE_PRESERVED_AUTH_VARS
    if backend == "codex":
        return _CODEX_PRESERVED_AUTH_VARS
    if backend == "opencode":
        return _OPENCODE_PRESERVED_AUTH_VARS
    raise ValueError(f"unknown backend for preserve-env: {backend!r}")


# Timeout (seconds) for the cross-user kill subprocess on the wrap
# path. Matches the persistent backend's `_async_sudo_kill` cap. A
# hung kill must not stall the reasoner's own timeout cleanup, so
# the cap is intentionally short relative to memory_extraction's
# stage-1 / stage-2 timeouts.
_CROSS_USER_KILL_TIMEOUT_S: float = 5.0


# ── Shared wrap / kill helpers ──────────────────────────────────────


def _wrap_cmd_for_user(cmd: list[str], target_user: str, backend: str) -> list[str]:
    """
    Prefix `cmd` with `sudo -H -u <target> --preserve-env=<csv> --`.

    `-H` rewrites HOME so the agent reads its config and OAuth state
    under the target user's home. The `--preserve-env=<csv>` clause
    lets the auth variables in `_preserved_auth_vars_for(backend)`
    survive sudo's `env_reset`; the per-os_user sudoers entries
    generated by `install._generate_sudoers` carry the `SETENV:` tag
    that authorizes the passthrough. PATH is not in the preserve list
    because the bot user's PATH is already in the subprocess env via
    the per-backend allow-list, and sudo resolves bare commands like
    `claude` / `codex` against that PATH before applying its sudoers
    check (the sudoers rule pins the resolved absolute path).
    """
    preserve = ",".join(_preserved_auth_vars_for(backend))
    return [
        "sudo",
        "-H",
        "-u",
        target_user,
        f"--preserve-env={preserve}",
        "--",
        *cmd,
    ]


async def _kill_target_user_tree(
    *,
    target_user: str,
    pgid: int,
    purpose: str,
    backend: str,
) -> None:
    """
    Send SIGKILL to every target-user process in the spawn's group.

    The sudo wrap leaves the bot holding the `sudo` process while
    the agent (and any descendants like the npm-wrapped codex's
    node/Rust child pair) runs under `target_user`. POSIX
    permission rules prevent the service user from signaling those
    descendants directly; the workaround is `sudo -n -u <target>
    /bin/kill -KILL -<pgid>` which the target user CAN run
    (authorized by the per-os_user `/bin/kill` rule
    `_generate_sudoers` already emits). The leading `-` on the
    PID arg makes kill's target a process group; sending to the
    negative group id signals every process the target user has
    permission to signal in that group, covering descendants at
    every tree depth without per-PID discovery.

    Bounded by `_CROSS_USER_KILL_TIMEOUT_S` so a hung sudo or
    missing `/bin/kill` does not stall the reasoner's own timeout
    cleanup. Failures (non-zero rc, timeout, FileNotFoundError on
    sudo itself) are logged and swallowed; the caller still reaps
    the wrapper afterward, accepting the orphan risk for that
    deployment edge case.
    """
    cmd = ["sudo", "-n", "-u", target_user, "/bin/kill", "-KILL", f"-{pgid}"]
    try:
        kill_proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        log.warning(
            "cross-user kill spawn failed: sudo not found (purpose=%s backend=%s target=%s pgid=%d)",
            purpose,
            backend,
            target_user,
            pgid,
        )
        return
    try:
        _stdout, stderr = await asyncio.wait_for(kill_proc.communicate(), timeout=_CROSS_USER_KILL_TIMEOUT_S)
    except TimeoutError:
        kill_proc.kill()
        with contextlib.suppress(BaseException):
            await kill_proc.wait()
        log.warning(
            "cross-user kill timed out (purpose=%s backend=%s target=%s pgid=%d)",
            purpose,
            backend,
            target_user,
            pgid,
        )
        return
    rc = kill_proc.returncode if kill_proc.returncode is not None else -1
    if rc != 0:
        log.warning(
            "cross-user kill returned rc=%d (purpose=%s backend=%s target=%s pgid=%d stderr=%r)",
            rc,
            purpose,
            backend,
            target_user,
            pgid,
            stderr[:200],
        )


def _os_user_log_field(effective_user: str | None) -> str:
    """
    Render the `os_user=...` log field value.

    `<target>` when the wrap is active, `self` when not (the bot
    process user is the agent's UID). Centralized so every emit
    site uses the same vocabulary.
    """
    return effective_user if effective_user else "self"


# ── Shared schema-payload parsing ────────────────────────────────────


# Matches one fenced code block and captures its body. The info
# string after the opening fence (`json`, `JSON`, or anything else)
# is consumed but not constrained: models label payload fences
# inconsistently and the json.loads attempt on the body is the real
# filter. DOTALL lets the body span newlines; the non-greedy body
# keeps back-to-back fences in one response as separate matches
# instead of one block swallowing everything between the first
# opener and the last closer.
_FENCED_BLOCK_RE = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)


def _parse_schema_payload(text: str) -> Any:
    """
    Parse a schema-backed one-shot response out of a model's chat text.

    Claude one-shots never need this leniency (`--json-schema` makes
    the CLI emit validated JSON) and codex rarely does
    (`--output-schema` shapes the final message CLI-side), but
    opencode has no structured-output channel at all: the JSON-only
    instruction rides inside the prompt, so compliance is pure model
    discipline. Models that narrate before answering wrap the payload
    in chat noise (observed with deepseek-v4-flash: a reasoning
    preamble followed by a ```json fence), and a bare json.loads on
    that text rejects an otherwise valid payload. Both the opencode
    and codex schema paths parse through this helper so the two
    reasoners keep mirroring each other.

    Three tiers, cheapest first:

    1. Bare parse of the stripped text. Preserves the strict-path
       behavior exactly, including non-dict results (a bare scalar or
       list still reaches the caller's non_object_json check rather
       than being rejected here).
    2. Fenced blocks: each block body that parses as a JSON object is
       a candidate.
    3. Brace scan: `raw_decode` at each `{` pulls an object out of
       unfenced prose.

    Within tiers 2 and 3 the LAST candidate wins: models think first
    and answer last, so when a preamble quotes an example object the
    payload that follows it is the real answer.

    Raises the tier-1 `json.JSONDecodeError` when no tier produces an
    object, so call sites keep their existing except shape and the
    logged error carries the position of the original parse failure.
    """
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        bare_error = exc

    payload: Any = None
    for match in _FENCED_BLOCK_RE.finditer(text):
        try:
            candidate = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            payload = candidate
    if payload is not None:
        return payload

    # Tier 3 walks every `{` because a failed raw_decode reports
    # nothing about where a viable object might start. On success the
    # scan resumes AFTER the parsed object: nested objects are part
    # of the outer candidate, and re-parsing them would let an inner
    # fragment shadow the complete payload. Quadratic on brace-dense
    # garbage in the worst case, but one-shot responses are a few KB
    # and this tier only runs after both cheaper tiers missed.
    decoder = json.JSONDecoder()
    search_from = 0
    while True:
        brace = text.find("{", search_from)
        if brace == -1:
            break
        try:
            candidate, end = decoder.raw_decode(text, brace)
        except json.JSONDecodeError:
            search_from = brace + 1
            continue
        payload = candidate
        search_from = end
    if payload is not None:
        return payload

    raise bare_error


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
    - No cost cap is passed: subscription auth has no real per-token
      cost. Runaway-loop protection comes from
      `asyncio.wait_for(timeout)` instead.
    - `--permission-mode bypassPermissions` is acceptable because
      `--tools ""` leaves nothing to permit or deny.

    Payload goes on stdin, not argv. Argv is visible via `ps -ef` and
    often captured by process accounting; stdin is not world-readable
    and survives a future multi-user transition without code changes.
    """

    def __init__(self, *, cwd: Path | None = None, os_user: str | None = None) -> None:
        """
        Args:
            cwd: Override for the subprocess working directory. Tests
                pass a temp path so the run does not touch the real
                DATA_DIR. Production callers leave this unset; the
                reasoner uses `_EXTRACTOR_CWD` and ensures it exists
                on first call.
            os_user: Optional OS user to run claude as via `sudo -H -u`.
                When None or matching the bot process user (the
                self-sudo-skip case, detected by `resolve_claude_user`),
                claude spawns directly with no wrap, preserving the
                existing single-user behavior. When set to a different
                target, the wrap is applied with `--preserve-env`
                carrying the auth vars from `_preserved_auth_vars_for`.
        """
        self._cwd = cwd if cwd is not None else _EXTRACTOR_CWD
        self._os_user = os_user

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
        # Lazy mkdir + chmod: deferred from import time on purpose so
        # a permission failure here surfaces as a logged miss rather
        # than crashing the bot at startup. The chmod runs every
        # call so a pre-existing tighter mode self-heals on the next
        # extraction (cross-user routing needs the cwd world-
        # traversable for the target user to enter it).
        self._cwd.mkdir(parents=True, exist_ok=True)
        self._cwd.chmod(0o755)

        # Resolve the claude binary through the shared resolver so
        # config validation, this argv, and the smoke output all see
        # the same resolution result. BinaryResolutionError from the
        # resolver becomes OneShotRoutingError here so the existing
        # memory_extraction `except OneShotError` catch surface still
        # collapses the failure to the zero-state extraction result;
        # the rewrap preserves the leaf-resolver / reasoner-error
        # split documented on oneshot_binary.
        try:
            resolved_binary = resolve_oneshot_binary("claude")
        except BinaryResolutionError as e:
            raise OneShotRoutingError(str(e)) from e

        cmd: list[str] = [resolved_binary, "--print"]
        if model is not None:
            cmd.extend(["--model", model])
        cmd.extend(["--output-format", "json"])
        if json_schema is not None:
            # `claude --print --json-schema <schema>` validates the
            # model's structured-output payload CLI-side. Stage 1 and
            # stage 2 both rely on this defense-in-depth gate; the
            # caller's Python validation is the primary contract, but
            # losing the CLI gate would let more malformed payloads
            # reach the parser.
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

        # Per-user OS routing. resolve_claude_user returns None when
        # the target matches the bot process user (the self-sudo-skip
        # case from PR #194) or when no target was supplied at all;
        # either way the direct spawn path is byte-identical to the
        # pre-routing behavior. Claude memory deliberately allows the
        # None case so existing Max-plan installs that never set per-
        # user os_user keep working without configuration changes.
        effective_user = resolve_claude_user(self._os_user)
        if effective_user is not None:
            cmd = _wrap_cmd_for_user(cmd, effective_user, "claude")

        # Allow-listed env: only forward vars from
        # _SUBPROCESS_ENV_ALLOWLIST that are present in the parent
        # environment. PATH is allow-listed so sudo can resolve the
        # bare `claude` invocation against the bot user's PATH; the
        # sudoers rule pins the resolved absolute path so the
        # passthrough is bounded.
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
            # New session on the wrap path so kill -KILL -<pgid> hits
            # every target-user descendant; default semantics on the
            # direct path.
            start_new_session=bool(effective_user),
        )
        os_user_field = _os_user_log_field(effective_user)
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=prompt.encode("utf-8")),
                timeout=timeout,
            )
        except TimeoutError:
            # Wrap path: escalate via the target user's permission to
            # signal its own process group before reaping the sudo
            # wrapper, so the agent (and any descendants) does not
            # outlive the wrapper. Direct path: same kill-and-await
            # pattern as before, no escalation needed since the agent
            # IS the proc.
            if effective_user is not None:
                await _kill_target_user_tree(
                    target_user=effective_user,
                    pgid=proc.pid,
                    purpose=purpose,
                    backend="claude",
                )
            proc.kill()
            await proc.wait()
            duration_ms = int((time.monotonic() - start) * 1000)
            log.info(
                "oneshot_reasoner purpose=%s backend=claude model=%s duration_ms=%d outcome=timeout error_category=timeout os_user=%s",
                purpose,
                model,
                duration_ms,
                os_user_field,
            )
            raise OneShotTimeout() from None

        duration_ms = int((time.monotonic() - start) * 1000)
        returncode = proc.returncode if proc.returncode is not None else -1
        if returncode != 0:
            log.info(
                "oneshot_reasoner purpose=%s backend=claude model=%s duration_ms=%d outcome=subprocess_error error_category=non_zero_exit returncode=%d os_user=%s",
                purpose,
                model,
                duration_ms,
                returncode,
                os_user_field,
            )
            raise OneShotSubprocessError(returncode=returncode, stderr=stderr)

        log.info(
            "oneshot_reasoner purpose=%s backend=claude model=%s duration_ms=%d outcome=success returncode=0 os_user=%s",
            purpose,
            model,
            duration_ms,
            os_user_field,
        )
        # raw_metadata: subprocess-layer data only. NO JSON envelope
        # parse here - that is memory_extraction's contract. Future
        # provider implementations may populate backend-specific
        # metadata, but current callers depend on nothing beyond
        # what is captured below.
        return OneShotResult(
            text=stdout.decode("utf-8", errors="replace"),
            backend="claude",
            model=model,
            raw_metadata={
                "returncode": returncode,
                "stderr": stderr,
                "cwd": str(self._cwd),
                # cmd is the post-wrap argv (includes sudo prefix on
                # cross-user routing); resolved_binary is the pre-wrap
                # agent path. Smoke prints resolved_binary directly so
                # the operator-visible "which binary ran" answer does
                # not regress under the sudo wrap, where cmd[0] is
                # "sudo" rather than the agent.
                "cmd": list(cmd),
                "resolved_binary": resolved_binary,
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


# JSON Schema keywords OpenAI's strict structured-outputs rejects.
# Codex `exec --output-schema <FILE>` forwards the schema verbatim to
# the Chat Completions endpoint in strict mode, so any of these in the
# schema produce a 400 `invalid_json_schema` and the codex subprocess
# exits non-zero. The list is what OpenAI's strict-mode docs disallow
# as of the codex CLI version verified during the #498 eval-gate work
# on 2026-05-19; future loosening on the OpenAI side would only mean
# this strip becomes redundant, never that it does the wrong thing.
_STRICT_MODE_DISALLOWED_KEYS: frozenset[str] = frozenset(
    {
        "minLength",
        "maxLength",
        "pattern",
        "format",
        "minimum",
        "maximum",
        "multipleOf",
        "minItems",
        "maxItems",
        "uniqueItems",
        "default",
        # `not` composition is currently unsupported by OpenAI strict
        # structured outputs even though plain JSON Schema allows it.
        # The production memory schemas never use `not`; stripping it
        # is forward-protection against a future schema author adding
        # it without knowing the codex CLI would 400 on it.
        "not",
    }
)


def _sanitize_for_codex(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Transform a JSON Schema into OpenAI strict structured-outputs form.

    OpenAI strict mode is a tighter subset of JSON Schema than what
    Claude's `--json-schema` accepts. Three rules apply, derived from
    the API's error responses against the production memory schemas:

    1. `additionalProperties: false` on every object node. The kai
       schemas already set this; the sanitizer is defensive in case a
       future schema author forgets.
    2. `required` must include every key in `properties`. Truly
       optional properties (e.g., `confirmation_quote` on confirmed-
       action facts, `existing_id` on consolidation intents, `lessons`
       on episodes) violate this. The sanitizer adds every property
       to `required` and widens the `type` of the previously-optional
       properties to include `"null"`, so the model can emit `null`
       when the field does not apply.
    3. Many validators are disallowed: `minLength`, `maxLength`,
       `pattern`, `format`, `minimum`, `maximum`, `multipleOf`,
       `minItems`, `maxItems`, `uniqueItems`, `default`. The
       sanitizer strips these recursively from every node.

    Allowed keywords are preserved: `type`, `properties`, `required`,
    `items`, `enum`, `additionalProperties`, `description`, `oneOf`,
    `anyOf`. `not` is NOT preserved; OpenAI strict structured outputs
    currently rejects it even though plain JSON Schema accepts it.

    Why sanitize rather than maintain a parallel codex-only schema:
    the bound constraints (string lengths, value ranges, array sizes)
    are already enforced post-extraction by the Python validators in
    `kai.memory_extraction` (`_validate_facts`, `_validate_episode`).
    Dropping them from the schema does not widen the production
    contract; it only relaxes what the model is INSTRUCTED to produce.
    The caller's bound checks continue to gate storage.

    Pure function. The input schema is not mutated; the output is a
    fresh dict tree the caller can write to disk and discard.
    """
    return _strict_node(schema)


def _strict_node(node: Any) -> Any:
    """Recursive worker for `_sanitize_for_codex`.

    Returns a sanitized copy of `node`. Lists and dicts are walked
    structurally; scalars round-trip unchanged. Object nodes (those
    with `type == "object"`, or with a `properties` field at the
    top level when `type` is absent) get the strict-mode treatment;
    other dicts (e.g., the `properties` mapping itself, which has
    property-name keys rather than schema keywords) are walked
    transparently.
    """
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in _STRICT_MODE_DISALLOWED_KEYS:
                continue
            out[key] = _strict_node(value)
        node_type = out.get("type")
        # An object node carries `type: "object"` AND a `properties`
        # mapping. The schema's top-level `properties` value is itself
        # a dict whose keys are property names; we recurse into each
        # property value (which IS a schema) but do not treat the
        # outer `properties` mapping as an object node.
        if node_type == "object" and isinstance(out.get("properties"), dict):
            properties: dict[str, Any] = out["properties"]
            if "additionalProperties" not in out:
                out["additionalProperties"] = False
            existing_required = list(out.get("required") or [])
            required_set = set(existing_required)
            for prop_name in properties:
                if prop_name in required_set:
                    continue
                # Previously-optional property: widen its type to
                # include `null` so the model can emit `null` for
                # absent values, then add it to required.
                properties[prop_name] = _widen_nullable(properties[prop_name])
                existing_required.append(prop_name)
                required_set.add(prop_name)
            out["required"] = existing_required
        return out
    if isinstance(node, list):
        return [_strict_node(item) for item in node]
    return node


def _widen_nullable(prop_schema: Any) -> Any:
    """Add `null` to the property's `type` so the model can emit null.

    Three cases the production schemas exercise:

    - `type: "string"` -> `type: ["string", "null"]`.
    - `type: ["string", "integer"]` (already a list) -> append "null"
      if not already present.
    - No `type` at all (e.g., a property whose schema is just an
      `enum` or an `oneOf`): leave unchanged. OpenAI strict mode may
      or may not accept this shape in the optional-via-required-and-
      null form; in practice the production schemas do not use it,
      and forcing `type: "null"` onto an enum-only property would be
      a guess about intent. If a future schema hits this, the API
      will return a 400 and the operator-side log review will surface
      the property name.
    """
    if not isinstance(prop_schema, dict):
        return prop_schema
    out = dict(prop_schema)
    existing_type = out.get("type")
    if existing_type is None:
        return out
    if isinstance(existing_type, list):
        if "null" not in existing_type:
            out["type"] = [*existing_type, "null"]
        return out
    if existing_type == "null":
        return out
    out["type"] = [existing_type, "null"]
    return out


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

    def __init__(self, *, cwd: Path | None = None, os_user: str | None = None) -> None:
        """
        Args:
            cwd: Override for the subprocess working directory and the
                schema temp file's parent. Tests pass a temp path so
                the run does not touch the real DATA_DIR. Production
                callers leave this unset; the reasoner uses
                `_EXTRACTOR_CWD` and ensures it exists on first call.
            os_user: Optional target OS user for the codex subprocess,
                symmetric with claude. `None` (or a value that
                `resolve_claude_user` collapses to the current process
                user) spawns codex in-process as the bot user - the
                self-sudo-skip path. A non-bot username wraps the
                argv in `sudo -H -u <user>` for cross-user isolation.
                Operators who want the cross-user separation set
                `os_user` to a different OS account in users.yaml;
                operators running kai under their own account leave
                it unset and codex runs in-process the same way
                claude does. The persistent codex chat backend in
                `src/kai/codex.py` already uses this resolution
                shape; this reasoner now follows it.
        """
        self._cwd = cwd if cwd is not None else _EXTRACTOR_CWD
        self._os_user = os_user

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
        # Lazy mkdir + chmod: deferred from import time so a
        # permission failure surfaces as a logged miss rather than
        # crashing the bot at startup. The unconditional chmod
        # self-heals a pre-existing tighter mode on the next call.
        self._cwd.mkdir(parents=True, exist_ok=True)
        self._cwd.chmod(0o755)

        # Resolve the codex binary through the shared resolver so
        # config validation, this argv, and the smoke output all see
        # the same resolution result. Tighter than the previous inline
        # `os.environ.get("CODEX_BIN") or "codex"` because the resolver
        # validates an explicit CODEX_BIN override as is-file plus
        # executable (no PATH fallback for a bad override), matching
        # config-load validation. BinaryResolutionError becomes
        # OneShotRoutingError here so the existing memory_extraction
        # `except OneShotError` catch surface is unchanged.
        try:
            resolved_binary = resolve_oneshot_binary("codex")
        except BinaryResolutionError as e:
            raise OneShotRoutingError(str(e)) from e
        cmd: list[str] = [
            resolved_binary,
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

        # Per-user OS routing target. resolve_claude_user returns
        # None when `os_user` is unset OR when it resolves to the
        # current process user (the self-sudo-skip path). Codex
        # treats None the same way claude does: spawn in-process as
        # the bot user, skipping the sudo wrap. A non-None target
        # produces the sudo-wrapped argv assembled below. Mirrors
        # the persistent codex chat backend's resolution shape so
        # both codex spawn paths share the same direct-vs-wrapped
        # decision logic.
        effective_user = resolve_claude_user(self._os_user)

        # Schema temp file lifecycle. Written before subprocess spawn
        # so the CLI sees a populated file; removed in `finally` so
        # success, timeout, non-zero exit, and output-parse failures
        # all clean up. `delete=False` is required because the
        # subprocess opens the file by path, not by inherited fd;
        # NamedTemporaryFile's auto-delete on close would yank the
        # file out from under the running codex process.
        #
        # Mode 0o644 is set explicitly after the file is populated:
        # NamedTemporaryFile defaults to 0o600 (bot-only readable),
        # which the target-user subprocess cannot open on the wrap
        # path. World-read is the right posture because the file is
        # an input the bot generates for the agent to consume; no
        # agent ever needs to write to it, and its content is the
        # JSON Schema the caller already supplied in argv-adjacent
        # form (not secret).
        schema_path: Path | None = None
        if json_schema is not None:
            # Sanitize before writing. Codex forwards the schema to
            # OpenAI's Chat Completions endpoint in strict structured-
            # outputs mode, which has tighter requirements than the
            # JSON Schema subset claude's `--json-schema` accepts.
            # See `_sanitize_for_codex` for the transformation rules.
            # The original `json_schema` reference is preserved for
            # the downstream required-fields check on the model's
            # response, which uses the caller's TRULY required list
            # (not the widened all-properties list the sanitizer
            # produces).
            sanitized_schema = _sanitize_for_codex(json_schema)
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                prefix="codex-schema-",
                dir=str(self._cwd),
                delete=False,
                encoding="utf-8",
            ) as fh:
                json.dump(sanitized_schema, fh)
                schema_path = Path(fh.name)
            schema_path.chmod(0o644)
            cmd.extend(["--output-schema", str(schema_path)])

        # Wrap codex in `sudo -H -u <target>` with the auth preserve
        # list when running cross-user. When `effective_user is None`
        # (same-user spawn: os_user unset OR matches the bot user),
        # the argv stays direct - codex runs in-process as the bot
        # user, the same shape the persistent codex chat backend
        # uses for self-sudo-skip. The `start_new_session` flag and
        # the timeout-escalation branch below already gate on the
        # same predicate so the cross-user path stays unchanged.
        if effective_user is not None:
            cmd = _wrap_cmd_for_user(cmd, effective_user, "codex")

        # Allow-listed env: only forward keys present in the parent
        # env. Defense-in-depth against a future regression that
        # tries to reuse Claude's env list. PATH is allow-listed so
        # sudo can resolve the bare `codex` invocation against the
        # bot user's PATH; CODEX_BIN takes precedence when set, in
        # which case the absolute path goes straight into argv.
        subprocess_env: dict[str, str] = {key: os.environ[key] for key in _CODEX_ENV_ALLOWLIST if key in os.environ}

        stdin_payload = _render_codex_stdin(system_prompt, prompt)
        os_user_field = _os_user_log_field(effective_user)

        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._cwd),
                env=subprocess_env,
                start_new_session=bool(effective_user),
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=stdin_payload.encode("utf-8")),
                    timeout=timeout,
                )
            except TimeoutError:
                # Wrap path: escalate via the target user's permission
                # to signal its own process group (kill -KILL -<pgid>)
                # so the entire descendant tree (sudo -> node ->
                # codex on npm-wrapped installs, or sudo -> codex on
                # a direct binary install) terminates before the
                # wrapper is reaped. Direct path keeps the original
                # kill+await.
                if effective_user is not None:
                    await _kill_target_user_tree(
                        target_user=effective_user,
                        pgid=proc.pid,
                        purpose=purpose,
                        backend="codex",
                    )
                proc.kill()
                await proc.wait()
                duration_ms = int((time.monotonic() - start) * 1000)
                log.info(
                    "oneshot_reasoner purpose=%s backend=codex model=%s duration_ms=%d outcome=timeout error_category=timeout os_user=%s",
                    purpose,
                    model,
                    duration_ms,
                    os_user_field,
                )
                raise OneShotTimeout() from None

            duration_ms = int((time.monotonic() - start) * 1000)
            returncode = proc.returncode if proc.returncode is not None else -1
            if returncode != 0:
                log.info(
                    "oneshot_reasoner purpose=%s backend=codex model=%s duration_ms=%d outcome=subprocess_error error_category=non_zero_exit returncode=%d os_user=%s",
                    purpose,
                    model,
                    duration_ms,
                    returncode,
                    os_user_field,
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
                    "oneshot_reasoner purpose=%s backend=codex model=%s duration_ms=%d outcome=success returncode=0 os_user=%s",
                    purpose,
                    model,
                    duration_ms,
                    os_user_field,
                )
                return OneShotResult(
                    text=final_text,
                    backend="codex",
                    model=model,
                    raw_metadata={
                        "returncode": returncode,
                        "stderr": stderr,
                        "cwd": str(self._cwd),
                        # cmd is post-wrap (includes sudo prefix on
                        # cross-user routing); resolved_binary is the
                        # pre-wrap agent path. Smoke prints
                        # resolved_binary so the operator-visible
                        # "which binary ran" answer survives the sudo
                        # wrap, where cmd[0] is "sudo" rather than
                        # the agent.
                        "cmd": list(cmd),
                        "resolved_binary": resolved_binary,
                    },
                    duration_ms=duration_ms,
                )

            # Schema-backed path. The final agent_message must carry
            # a JSON object; `_parse_schema_payload` tolerates fence /
            # prose wrapping around it, and anything it cannot recover
            # is OneShotOutputError so memory_extraction's caller-side
            # mapping collapses it to the zero-state extraction result
            # instead of letting a malformed payload reach the fact
            # validator.
            if not final_text:
                log.info(
                    "oneshot_reasoner purpose=%s backend=codex model=%s duration_ms=%d outcome=output_error error_category=empty_agent_message returncode=0 os_user=%s",
                    purpose,
                    model,
                    duration_ms,
                    os_user_field,
                )
                raise OneShotOutputError("codex produced no final agent_message")
            try:
                payload = _parse_schema_payload(final_text)
            except json.JSONDecodeError as exc:
                log.info(
                    "oneshot_reasoner purpose=%s backend=codex model=%s duration_ms=%d outcome=output_error error_category=invalid_json returncode=0 os_user=%s",
                    purpose,
                    model,
                    duration_ms,
                    os_user_field,
                )
                raise OneShotOutputError(f"codex final text did not contain a JSON object: {exc}") from None

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
                    "oneshot_reasoner purpose=%s backend=codex model=%s duration_ms=%d outcome=output_error error_category=non_object_json returncode=0 os_user=%s",
                    purpose,
                    model,
                    duration_ms,
                    os_user_field,
                )
                raise OneShotOutputError("codex final JSON was not an object")

            required_fields = json_schema.get("required")
            if isinstance(required_fields, list):
                missing = [field for field in required_fields if field not in payload]
                if missing:
                    log.info(
                        "oneshot_reasoner purpose=%s backend=codex model=%s duration_ms=%d outcome=output_error error_category=missing_required_fields returncode=0 os_user=%s",
                        purpose,
                        model,
                        duration_ms,
                        os_user_field,
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
                "oneshot_reasoner purpose=%s backend=codex model=%s duration_ms=%d outcome=success returncode=0 os_user=%s",
                purpose,
                model,
                duration_ms,
                os_user_field,
            )
            return OneShotResult(
                text=envelope,
                backend="codex",
                model=model,
                raw_metadata={
                    "returncode": returncode,
                    "stderr": stderr,
                    "cwd": str(self._cwd),
                    # cmd / resolved_binary as documented on the
                    # free-form return above; same fields populate
                    # on every codex success branch so the smoke
                    # output reads the same shape regardless of
                    # whether json_schema was set.
                    "cmd": list(cmd),
                    "resolved_binary": resolved_binary,
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


# ── OpenCode implementation ─────────────────────────────────────────


# Env vars forwarded to the opencode one-shot subprocess. Distinct from
# the claude / codex allow-lists because opencode does not consume
# CLAUDE_CONFIG_DIR or CODEX_HOME; it has its own auth state under
# ~/.local/share/opencode/. The provider API keys mirror
# `_OPENCODE_PRESERVED_AUTH_VARS` exactly because the same vars that
# survive sudo's env_reset are the ones the in-process spawn also
# needs (no-wrap path is the bot user; cross-user path is the target
# user with the same per-provider key contract). PATH is required so
# opencode resolves its own bundled helpers and any subprocesses it
# spawns. HOME is required so opencode finds `~/.local/share/opencode/
# auth.json`; on the wrap path `sudo -H` rewrites HOME to the target
# user automatically, on the no-wrap path the bot user's HOME is the
# right value already.
_OPENCODE_ENV_ALLOWLIST = (
    "PATH",
    "HOME",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
    "DEEPSEEK_API_KEY",
)


# Hard handshake timeout for `initialize` and `session/new`. The
# conversational backend uses `self.timeout_seconds` for the same
# reads but that value is the per-turn cap (often 600+s for long
# reasoning workloads). Handshake is a separate failure surface: a
# stalled handshake means opencode never produced a valid initialize
# response, so spending 10 minutes waiting on it is just dead time
# before the reasoner fails the call. 30 seconds is generous for a
# subprocess that normally finishes the handshake in under a second.
_OPENCODE_HANDSHAKE_TIMEOUT_S: int = 30


def _render_opencode_session_prompt(system_prompt: str | None, prompt: str) -> str:
    """
    Build a boundary-delimited session/prompt payload for an opencode
    one-shot call.

    The ACP `session/prompt` shape has no system-prompt slot. Round-2
    smoke against opencode 1.15.11 confirmed that injecting an agent
    definition through OPENCODE_CONFIG_CONTENT validates without
    taking effect on the model's behavior, and `--agent` falls back
    to the default agent on a missing-name. The remaining channel
    that the model reliably attends to is the user prompt itself.

    The render mirrors `_render_codex_stdin`: prepend the system text
    inside a randomized boundary block, then append the user prompt
    below it. The boundary token is `secrets.token_hex(4)` (32 bits)
    via `make_boundary`. That entropy is collision-avoidance, NOT a
    security barrier against adversarial prompts; the value matches
    the codex reasoner exactly so the two boundary helpers stay
    symmetric and a future widening can flow through `make_boundary`
    in one place.

    A None or empty `system_prompt` returns the user prompt unchanged
    rather than emitting a free-floating boundary block with no
    content; matches the no-system-prompt semantics callers already
    get from claude's `--system-prompt` flag being omitted.
    """
    if not system_prompt:
        return prompt
    begin, end = make_boundary("SYSTEM")
    return f"{begin}\n{system_prompt}\n{end}\n\n{prompt}"


class OpenCodeOneShotReasoner:
    """
    Spawns `opencode acp` once per call and returns either the raw
    accumulated agent_message text or, when a JSON schema is supplied,
    a normalized `{"is_error": false, "structured_output": ...}`
    envelope string that matches the shape memory_extraction.py
    already parses.

    OpenCode differs from claude and codex on every axis that matters:
    no CLI one-shot mode that reliably emits text (round-2 smoke
    against opencode 1.15.11 showed `opencode run --format json`
    emits a `text` event on stdout for only two of twelve smoke
    invocations; the other ten exited rc=0 with no text on the wire
    while the model actually produced output internally), so the
    transport contract is the same JSON-RPC-over-stdio ACP layer the
    conversational backend uses. Each call spawns a fresh
    `opencode acp` subprocess, drives the handshake, sends one
    `session/prompt` request, accumulates response text from
    `session/update` notifications, denies any tool-permission
    request that appears mid-stream, and tears the subprocess down.

    The transport reuses the module-level `write_rpc` and
    `read_result` free functions from `kai.acp`, plus
    `extract_opencode_text_delta` and
    `handle_opencode_permission_request` from `kai.opencode`. The
    permission policy passed in is `"reject_once"`; the
    conversational backend uses `"allow_always"` against the same
    free function so tool calls can run in chat. The split is
    deliberate: memory extraction / PR review / issue triage
    prompts must NOT execute tools, but the rejection is scoped to
    the single one-shot invocation rather than persisted to user
    config.

    Model selection flows through `OPENCODE_CONFIG_CONTENT`. The
    CLI accepts no `--model` flag; opencode reads inline JSON config
    from the env var at startup. Model strings use OpenCode's full
    `provider/model` shape (validated against `is_opencode_model_shape`
    at config-load time so a malformed entry never reaches this
    reasoner). When `model is None`, opencode falls back to whatever
    its own config files specify, matching the conversational
    backend's behavior.

    System prompt injection: ACP and the `session/prompt` shape do
    not expose a system-prompt slot, so the system text is prepended
    to the user prompt via `_render_opencode_session_prompt` with a
    randomized boundary block (mirroring `_render_codex_stdin`).

    Per-OS-user routing: mirrors claude and codex. When
    `os_user is None` or resolves to the bot user, opencode spawns
    directly. A non-bot target wraps the argv in
    `sudo -H -u <user> --preserve-env=<csv>` carrying the auth vars
    from `_preserved_auth_vars_for("opencode")`. The sudoers rule
    emitted by `install._generate_sudoers` for the opencode binary
    authorizes the passthrough; without that rule the wrap path
    fails with "a password is required" on the first call.

    Subprocess cleanup runs on every exit path (success, timeout,
    error, parse failure). The persistent backend's `_kill` /
    `shutdown` pattern is the model: send SIGKILL, await up to 5s,
    null all process references. The one-shot reasoner uses a
    bounded shutdown helper kept inside `run()` so the cleanup
    sequence stays in one place with the spawn it manages.
    """

    def __init__(self, *, cwd: Path | None = None, os_user: str | None = None) -> None:
        """
        Args:
            cwd: Override for the subprocess working directory. Tests
                pass a temp path so the run does not touch the real
                DATA_DIR. Production callers leave this unset; the
                reasoner uses `_EXTRACTOR_CWD` and ensures it exists
                on first call.
            os_user: Optional target OS user, symmetric with claude
                and codex. `None` (or a value that
                `resolve_claude_user` collapses to the current
                process user) spawns opencode in-process as the bot
                user; the self-sudo-skip path is identical to the
                no-os_user case. A non-bot username wraps the argv
                in `sudo -H -u <user>` with the opencode preserve
                list.
        """
        self._cwd = cwd if cwd is not None else _EXTRACTOR_CWD
        self._os_user = os_user

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
        # Lazy mkdir + chmod: deferred from import time so a permission
        # failure surfaces as a logged miss rather than crashing the bot
        # at startup. The unconditional chmod self-heals a pre-existing
        # tighter mode (the cross-user routing target must be able to
        # enter the cwd as a working directory).
        self._cwd.mkdir(parents=True, exist_ok=True)
        self._cwd.chmod(0o755)

        # Resolve the opencode binary through the shared resolver so
        # config validation, this argv, and the smoke output all see
        # the same resolution result. BinaryResolutionError becomes
        # OneShotRoutingError here so the existing memory_extraction
        # `except OneShotError` catch surface is unchanged; the
        # rewrap preserves the leaf-resolver / reasoner-error split
        # documented on oneshot_binary.
        try:
            resolved_binary = resolve_oneshot_binary("opencode")
        except BinaryResolutionError as e:
            raise OneShotRoutingError(str(e)) from e

        cmd: list[str] = [resolved_binary, "acp"]

        # Per-user OS routing. Same shape claude and codex use:
        # `resolve_claude_user` returns None when the target is unset
        # OR matches the bot user (self-sudo-skip), and the direct
        # spawn path is byte-identical to a no-os_user call. A
        # non-None target wraps the argv in sudo with the opencode
        # preserve list.
        effective_user = resolve_claude_user(self._os_user)
        if effective_user is not None:
            cmd = _wrap_cmd_for_user(cmd, effective_user, "opencode")

        # Allow-listed env: only forward keys present in the parent
        # env. Defense-in-depth against a future regression that tries
        # to reuse claude's or codex's env list. PATH is allow-listed
        # so sudo can resolve the bare `opencode` invocation against
        # the bot user's PATH; the sudoers rule pins the resolved
        # absolute path so the passthrough is bounded.
        subprocess_env: dict[str, str] = {key: os.environ[key] for key in _OPENCODE_ENV_ALLOWLIST if key in os.environ}
        # Model selection flows through OPENCODE_CONFIG_CONTENT. The
        # conversational backend uses the same env-driven mechanism
        # in `OpenCodeBackend.build_env`. When model is None the env
        # var is left out so opencode falls back to whatever its
        # config files specify, matching the conversational shape.
        if model is not None:
            subprocess_env["OPENCODE_CONFIG_CONTENT"] = json.dumps({"model": model})

        rendered_prompt = _render_opencode_session_prompt(system_prompt, prompt)
        os_user_field = _os_user_log_field(effective_user)

        start = time.monotonic()
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._cwd),
                env=subprocess_env,
                # New session on the wrap path so a future
                # `_kill_target_user_tree` on this reasoner has a pgid
                # to target. Direct path keeps default semantics so a
                # SIGKILL on `proc` reaches the agent alone.
                start_new_session=bool(effective_user),
            )

            # Drain stderr in the background so the pipe buffer cannot
            # fill and deadlock the subprocess. Discarded; opencode's
            # diagnostic output is not load-bearing for the one-shot
            # path. The conversational backend's `_drain_stderr`
            # surfaces WARNING-token lines, but the one-shot reasoner
            # is short-lived and the operator-readable failure surface
            # is the OneShotError typed exceptions, not stderr scrape.
            stderr_task = asyncio.create_task(_drain_proc_stderr(proc))

            # Run the full one-shot exchange inside the overall timeout
            # so timeout / handshake-error / response-parse failures
            # all map to typed errors with the elapsed time captured.
            # `asyncio.wait_for(timeout=None)` is a no-op timeout when
            # the caller did not specify one, matching the claude /
            # codex paths' identical semantics.
            try:
                result = await asyncio.wait_for(
                    self._drive_exchange(
                        proc=proc,
                        rendered_prompt=rendered_prompt,
                        purpose=purpose,
                        model=model,
                        json_schema=json_schema,
                    ),
                    timeout=timeout,
                )
            except TimeoutError:
                duration_ms = int((time.monotonic() - start) * 1000)
                log.info(
                    "oneshot_reasoner purpose=%s backend=opencode model=%s duration_ms=%d outcome=timeout error_category=timeout os_user=%s",
                    purpose,
                    model,
                    duration_ms,
                    os_user_field,
                )
                raise OneShotTimeout() from None

            accumulated, completion_error = result
            duration_ms = int((time.monotonic() - start) * 1000)

            if completion_error is not None:
                # JSON-RPC error response on the session/prompt id, OR
                # the subprocess exited mid-stream. Maps to
                # OneShotSubprocessError when we have a returncode
                # (process died) and OneShotOutputError when we have
                # only an error message (opencode itself reported it).
                if proc.returncode is not None:
                    log.info(
                        "oneshot_reasoner purpose=%s backend=opencode model=%s duration_ms=%d outcome=subprocess_error error_category=non_zero_exit returncode=%d os_user=%s",
                        purpose,
                        model,
                        duration_ms,
                        proc.returncode,
                        os_user_field,
                    )
                    raise OneShotSubprocessError(returncode=proc.returncode, stderr=b"")
                log.info(
                    "oneshot_reasoner purpose=%s backend=opencode model=%s duration_ms=%d outcome=output_error error_category=jsonrpc_error returncode=0 os_user=%s",
                    purpose,
                    model,
                    duration_ms,
                    os_user_field,
                )
                raise OneShotOutputError(f"opencode session/prompt error: {completion_error}")

            # Free-form path: hand the accumulated text back unchanged.
            # Memory extraction / review / triage all supply a schema,
            # so this branch exists only for any future free-form
            # caller. Claude and codex both keep the symmetric
            # branch; the one-shot family stays uniform on this.
            if json_schema is None:
                log.info(
                    "oneshot_reasoner purpose=%s backend=opencode model=%s duration_ms=%d outcome=success returncode=0 os_user=%s",
                    purpose,
                    model,
                    duration_ms,
                    os_user_field,
                )
                return OneShotResult(
                    text=accumulated,
                    backend="opencode",
                    model=model,
                    raw_metadata={
                        "returncode": 0,
                        "stderr": b"",
                        "cwd": str(self._cwd),
                        # cmd is post-wrap (includes the sudo prefix
                        # on cross-user routing); resolved_binary is
                        # the pre-wrap agent path. Smoke prints
                        # resolved_binary so the operator-visible
                        # "which binary ran" answer survives the sudo
                        # wrap, where cmd[0] is "sudo" rather than
                        # the agent.
                        "cmd": list(cmd),
                        "resolved_binary": resolved_binary,
                    },
                    duration_ms=duration_ms,
                )

            # Schema-backed path. Mirrors the codex reasoner exactly:
            # the accumulated text must carry a JSON object, with
            # `_parse_schema_payload` tolerating fence / prose wrapping
            # around it; anything it cannot recover is
            # OneShotOutputError so memory extraction's caller-side
            # mapping collapses it to the zero-state extraction result
            # instead of letting a malformed payload reach the fact
            # validator.
            if not accumulated:
                log.info(
                    "oneshot_reasoner purpose=%s backend=opencode model=%s duration_ms=%d outcome=output_error error_category=empty_response returncode=0 os_user=%s",
                    purpose,
                    model,
                    duration_ms,
                    os_user_field,
                )
                raise OneShotOutputError("opencode produced no accumulated text")
            try:
                payload = _parse_schema_payload(accumulated)
            except json.JSONDecodeError as exc:
                log.info(
                    "oneshot_reasoner purpose=%s backend=opencode model=%s duration_ms=%d outcome=output_error error_category=invalid_json returncode=0 os_user=%s",
                    purpose,
                    model,
                    duration_ms,
                    os_user_field,
                )
                raise OneShotOutputError(f"opencode accumulated text did not contain a JSON object: {exc}") from None

            if not isinstance(payload, dict):
                log.info(
                    "oneshot_reasoner purpose=%s backend=opencode model=%s duration_ms=%d outcome=output_error error_category=non_object_json returncode=0 os_user=%s",
                    purpose,
                    model,
                    duration_ms,
                    os_user_field,
                )
                raise OneShotOutputError("opencode accumulated JSON was not an object")

            required_fields = json_schema.get("required")
            if isinstance(required_fields, list):
                missing = [field for field in required_fields if field not in payload]
                if missing:
                    log.info(
                        "oneshot_reasoner purpose=%s backend=opencode model=%s duration_ms=%d outcome=output_error error_category=missing_required_fields returncode=0 os_user=%s",
                        purpose,
                        model,
                        duration_ms,
                        os_user_field,
                    )
                    raise OneShotOutputError(f"opencode accumulated JSON missing required fields: {missing}")

            # Wrap opencode's schema-shaped payload in the same
            # envelope claude emits natively (and codex now mirrors).
            # memory_extraction.py already walks the
            # `structured_output` field on a parsed dict; rewrapping
            # keeps the parser path provider-neutral.
            envelope = json.dumps({"is_error": False, "structured_output": payload})
            log.info(
                "oneshot_reasoner purpose=%s backend=opencode model=%s duration_ms=%d outcome=success returncode=0 os_user=%s",
                purpose,
                model,
                duration_ms,
                os_user_field,
            )
            return OneShotResult(
                text=envelope,
                backend="opencode",
                model=model,
                raw_metadata={
                    "returncode": 0,
                    "stderr": b"",
                    "cwd": str(self._cwd),
                    "cmd": list(cmd),
                    "resolved_binary": resolved_binary,
                },
                duration_ms=duration_ms,
            )
        finally:
            # Subprocess cleanup runs on every exit path: success,
            # timeout, subprocess error, output error, and any
            # unexpected exception propagating from the spawn itself.
            # The shutdown sequence sends SIGKILL on the wrap path
            # (after escalating through the target user's permission
            # to signal its own process group) and on the direct
            # path; the bounded `wait(timeout=5)` then reaps the
            # process. A second timeout means the OS has lost the
            # process; the function returns without raising because
            # the typed error path already populated the caller's
            # exception context.
            if proc is not None:
                await _shutdown_opencode_proc(proc, effective_user=effective_user, purpose=purpose)
                if "stderr_task" in locals():
                    stderr_task.cancel()
                    with contextlib.suppress(BaseException):
                        await stderr_task

    async def _drive_exchange(
        self,
        *,
        proc: asyncio.subprocess.Process,
        rendered_prompt: str,
        purpose: str,
        model: str | None,
        json_schema: dict[str, Any] | None,
    ) -> tuple[str, str | None]:
        """
        Run the ACP handshake + session/prompt + read loop on `proc`.

        Returns `(accumulated_text, completion_error)` where
        `completion_error` is None on success or the JSON-RPC error
        message string when opencode returned a matching-id error
        for the prompt. The caller maps `completion_error is not
        None` to the right typed exception (`OneShotOutputError`
        vs `OneShotSubprocessError` depending on whether the
        subprocess itself died).

        Handshake errors propagate as RuntimeError / TimeoutError
        from the shared `read_result` free function; the caller's
        try/finally wraps cleanup and the outer wait_for translates
        TimeoutError to `OneShotTimeout`.

        Kept as a separate method so the outer `run()` can wrap the
        whole exchange in a single `asyncio.wait_for(timeout=...)`
        and still hit the cleanup path uniformly.

        `purpose`, `model`, `json_schema` are accepted for symmetry
        with the broader call shape; they are not read here because
        the contract is "drive the bytes," not "decide success vs
        failure semantics." The caller owns those decisions after
        the exchange returns.
        """
        # Local request-id counter scoped to this short-lived
        # subprocess. Distinct from the conversational backend's
        # `self._next_id` because the one-shot has no persistent
        # state to keep across calls.
        next_id = 1

        # Step 1: initialize. Send request, await response. The
        # response shape is not consumed here; what matters is that
        # opencode acknowledges the protocol version before we issue
        # `session/new`.
        from kai import __version__

        next_id = await write_rpc(
            proc=proc,
            next_id=next_id,
            method="initialize",
            params={
                "protocolVersion": 1,
                "clientInfo": {"name": "kai", "version": __version__},
            },
        )
        await read_result(
            proc=proc,
            expected_id=1,
            timeout_seconds=_OPENCODE_HANDSHAKE_TIMEOUT_S,
            backend_label="OpenCode",
        )

        # Step 2: session/new. The cwd matches the subprocess working
        # directory so opencode's session model and Kai's view of the
        # working tree agree. mcpServers is empty: one-shot calls do
        # not need MCP integration, and excluding the field would
        # leave opencode reading whatever its config files specify
        # (potentially injecting tools we deny later anyway).
        next_id = await write_rpc(
            proc=proc,
            next_id=next_id,
            method="session/new",
            params={
                "cwd": str(self._cwd),
                "mcpServers": [],
            },
        )
        session_result = await read_result(
            proc=proc,
            expected_id=2,
            timeout_seconds=_OPENCODE_HANDSHAKE_TIMEOUT_S,
            backend_label="OpenCode",
        )
        session_id = session_result.get("sessionId")
        if not isinstance(session_id, str):
            # Defensive: opencode's session/new response should always
            # carry a string sessionId; missing it means the handshake
            # is corrupt and we cannot continue. Surface as a
            # RuntimeError so the outer try/finally cleans up the
            # subprocess; the caller maps unexpected exceptions to a
            # logged-and-collapsed extraction miss.
            raise RuntimeError(f"OpenCode session/new returned no sessionId: {session_result!r}")

        # Step 3: session/prompt. ACP wraps content in a list of
        # blocks; the one-shot only sends text content. The prompt id
        # is captured BEFORE write_rpc bumps the counter so the read
        # loop below can match against it.
        prompt_id = next_id
        next_id = await write_rpc(
            proc=proc,
            next_id=next_id,
            method="session/prompt",
            params={
                "sessionId": session_id,
                "prompt": [{"type": "text", "text": rendered_prompt}],
            },
        )

        # Step 4: read loop. Accumulate `agent_message_chunk` text;
        # reject any `session/request_permission` requests with the
        # `reject_once` policy; stop when the prompt response arrives
        # (success or error), draining any text chunks that flush
        # after the success response before returning. The loop has
        # no timeout of its own because the outer `asyncio.wait_for`
        # is the per-call cap (it also bounds the drain).
        accumulated = ""
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                # EOF: subprocess died mid-stream. Surface as a
                # completion_error so the caller maps to
                # OneShotSubprocessError (proc.returncode will be set
                # by the await above) or OneShotOutputError. The
                # accumulated text so far still goes back to the
                # caller in case a partial response is useful for
                # debugging (the typed-error path discards it, but
                # the contract is "what we got").
                return accumulated, "OpenCode process ended unexpectedly"

            try:
                msg = json.loads(line.decode())
            except json.JSONDecodeError:
                # Non-JSON lines from a misbehaving opencode build
                # (progress bars, debug prints) are skipped; the
                # shared `read_result` does the same for handshake
                # parsing.
                continue

            # Streaming notification (no `id` field). Accumulate text
            # via the shared free functions; skip every other
            # notification shape. `concat_opencode_text` handles the
            # sentence-boundary whitespace join the conversational
            # backend gets through `combine_text_chunks`; calling the
            # free function directly here keeps the one-shot path
            # symmetric without forcing it through the AcpBackend
            # hook surface.
            if "method" in msg and "id" not in msg:
                text = extract_opencode_text_delta(msg)
                if text:
                    accumulated = concat_opencode_text(accumulated, text)
                continue

            # Server-initiated request (both `method` and `id`). The
            # one-shot reasoner denies every permission request with
            # the `reject_once` policy; the response goes back on the
            # server's id so opencode unblocks and either re-prompts
            # or moves past the tool call.
            if "method" in msg and "id" in msg:
                server_id = msg["id"]
                response_body = handle_opencode_permission_request(msg, policy="reject_once")
                if response_body is not None:
                    payload = {"jsonrpc": "2.0", "id": server_id, "result": response_body}
                    line_out = (json.dumps(payload) + "\n").encode()
                    assert proc.stdin is not None
                    proc.stdin.write(line_out)
                    try:
                        await proc.stdin.drain()
                    except OSError:
                        # Stdin write failed; subprocess is gone.
                        # Surface as completion_error with the partial
                        # accumulated text so the caller can map to
                        # OneShotSubprocessError.
                        return accumulated, "OpenCode stdin closed mid-stream"
                continue

            # Final response for our session/prompt request.
            if msg.get("id") == prompt_id:
                if "error" in msg:
                    err = msg["error"].get("message", "unknown ACP error")
                    return accumulated, str(err)
                # Success. The result body is not consumed; the
                # accumulated text from session/update notifications
                # IS the answer. The response can beat the turn's
                # final text chunk(s) onto stdout, and the caller
                # kills the subprocess after this returns, so any
                # undrained tail would be silently lost; for the
                # JSON-consuming purposes (triage, extraction) a
                # missing tail makes the entire response unparseable.
                # See drain_late_text for the mechanism.
                accumulated = await drain_late_text(
                    proc=proc,
                    accumulated=accumulated,
                    extract_delta=extract_opencode_text_delta,
                    combine=concat_opencode_text,
                )
                return accumulated, None

            # Some other shape (a response to a request we did not
            # send, or a notification we already filtered). Skip and
            # keep reading.


async def _drain_proc_stderr(proc: asyncio.subprocess.Process) -> None:
    """
    Continuously read and discard stderr from a one-shot opencode
    subprocess.

    Without this, the stderr pipe buffer fills and the process can
    deadlock. Discarded (not logged at INFO) because the one-shot
    reasoner's operator-readable failure surface is the typed
    OneShotError family, not stderr scrape. A DEBUG line per drained
    string keeps the bytes traceable when an operator turns up
    logging without needing a code change.
    """
    if proc.stderr is None:
        return
    while True:
        try:
            line = await proc.stderr.readline()
        except Exception:
            return
        if not line:
            return
        text = line.decode(errors="replace").strip()
        if text:
            log.debug("OpenCode (one-shot) stderr: %s", text[:200])


async def _shutdown_opencode_proc(
    proc: asyncio.subprocess.Process,
    *,
    effective_user: str | None,
    purpose: str,
) -> None:
    """
    Bounded shutdown of an opencode one-shot subprocess on every
    exit path.

    Wrap path: escalate via the target user's `/bin/kill` permission
    to send SIGKILL to the spawn's process group first, then reap
    the sudo wrapper. The conversational claude / codex reasoners
    use the same pattern (`_kill_target_user_tree`); reusing that
    helper keeps the cross-user kill semantics uniform.

    Direct path: SIGKILL on `proc` directly. The 5-second
    `proc.wait(timeout=5)` matches the conversational
    AcpBackend.shutdown contract; a second timeout means the OS
    has lost the process and the caller's typed-error path has
    already populated the exception context, so the function
    returns silently rather than raising into the cleanup chain.
    """
    if proc.returncode is not None:
        # Already exited (typical happy path: the JSON-RPC response
        # arrived, the read loop returned, and opencode is in the
        # process of shutting itself down). Reap to avoid a zombie
        # and return.
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(proc.wait(), timeout=5)
        return

    if effective_user is not None:
        # Cross-user spawn: bot user does not have permission to
        # signal the target user's descendants directly. Use the
        # shared helper that hands the kill through `sudo /bin/kill`.
        with contextlib.suppress(BaseException):
            await _kill_target_user_tree(
                target_user=effective_user,
                pgid=proc.pid,
                purpose=purpose,
                backend="opencode",
            )

    # SIGKILL on the wrapper (or on the agent directly when no wrap).
    # OSError covers the race where the subprocess died between the
    # returncode check above and this kill.
    with contextlib.suppress(OSError):
        proc.kill()
    with contextlib.suppress(BaseException):
        await asyncio.wait_for(proc.wait(), timeout=5)
