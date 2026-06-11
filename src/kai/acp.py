"""
Shared ACP (Agent Client Protocol) subprocess backend layer.

Implements the AgentBackend ABC for any agent harness that speaks
JSON-RPC 2.0 over stdio in the ACP shape (initialize -> session/new
handshake; session/prompt with sessionId + prompt; session/update
streaming notifications; matching-id result/error responses).

Concrete adapters (GooseBackend, OpenCodeBackend) subclass AcpBackend
and override a narrow hook surface: command argv, env merge, init/
session-new params, stream notification parsing, completion / error
detection. Everything else - lock, lifecycle, context injection,
idle/response timeouts, stderr draining, restart / shutdown / force-
kill - lives here so two ACP backends cannot drift in their transport
behavior.

The ACP protocol:
    Startup:  initialize -> session/new (handshake)
    Input:    session/prompt (JSON-RPC request with sessionId + prompt)
    Output:   session/update (streaming notifications, no id field)
    Finish:   JSON-RPC result with matching id + stopReason

The hook surface:
    backend_name           : machine identifier ("goose", "opencode")
    backend_label          : human-readable label for error messages
    build_argv()           : subprocess command vector (may include model)
    build_env(base_env)    : layer backend-specific env onto base_env
    preserved_env_vars()   : env vars the cross-user sudo wrap preserves
    build_initialize_params() : params for the initialize JSON-RPC call
    build_session_new_params() : params for the session/new JSON-RPC call
    extract_session_id(result) : pull session ID from session/new result
    extract_text_delta(msg)   : pull user-visible assistant text from a
                                 session/update notification, or None
    is_completion(msg, id)    : True for a final-success response
    extract_error(msg, id)    : error message string or None
"""

import asyncio
import contextlib
import json
import logging
import os
import subprocess
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import kai
from kai.backend import (
    AgentBackend,
    AgentResponse,
    ApiContext,
    StreamEvent,
    apply_workspace_model,
    assemble_turn_context,
    build_foreign_workspace_reminder,
    build_session_context,
    ensure_user_memory,
    ensure_user_preferences,
)
from kai.config import DATA_DIR, WorkspaceConfig, parse_env_file, resolve_claude_user

log = logging.getLogger(__name__)


# Tokens that indicate an upstream-side rejection or schema failure
# in an ACP-server stderr line. When any of these substrings (case-
# insensitive) appears in a drained stderr line, the line is surfaced
# at WARNING instead of DEBUG. Set at module scope so concrete
# adapters cannot drift from the shared classifier.
_STDERR_WARNING_TOKENS: tuple[str, ...] = ("permission", "rejected", "invalid", "schema")


# ── Shared JSON-RPC wire primitives ───────────────────────────────


async def write_rpc(
    *,
    proc: asyncio.subprocess.Process | None,
    next_id: int,
    method: str,
    params: dict,
) -> int:
    """
    Write a JSON-RPC 2.0 request to the subprocess stdin and return the
    next request id.

    Pure wire primitive shared between the persistent `AcpBackend`
    (its `_write_rpc` instance method wraps this with `self._next_id`
    bookkeeping) and the one-shot reasoner in `kai.oneshot`
    (which holds its own private counter so its short-lived
    `opencode acp` session is independent from any conversational
    session). Both callers must thread the returned id back into their
    counter so the next request increments correctly.

    `proc` is typed Optional to accept the conversational backend's
    `self._proc` attribute directly (it is `None` between subprocess
    incarnations); the assert below makes the live-process invariant
    explicit and prevents a None-stdin crash with a typed message
    instead of an AttributeError.
    """
    # Both invariants here are caller-controlled; the assert is the
    # boundary that turns a programming error (calling without a live
    # process) into a clear typed failure rather than an opaque
    # AttributeError on `proc.stdin`.
    assert proc is not None and proc.stdin is not None
    request_id = next_id
    msg = (
        json.dumps(
            {
                "jsonrpc": "2.0",
                "method": method,
                "id": request_id,
                "params": params,
            }
        )
        + "\n"
    )
    proc.stdin.write(msg.encode())
    await proc.stdin.drain()
    return next_id + 1


async def read_result(
    *,
    proc: asyncio.subprocess.Process | None,
    expected_id: int,
    timeout_seconds: int,
    backend_label: str,
) -> dict:
    """
    Read stdout lines until a JSON-RPC result with `expected_id` appears.

    Discards session/update notifications (and any other lines that do
    not carry the expected id) that the harness may emit during
    startup. Raises `RuntimeError` on JSON-RPC error responses or
    process exit; raises `TimeoutError` when no line lands within
    `timeout_seconds`. Error messages are prefixed with
    `backend_label` so a startup failure log line identifies which
    adapter raised even when this free function is shared across
    backends.

    Shared between `AcpBackend._read_result` (conversational backends)
    and `OpenCodeOneShotReasoner.run` (one-shot reasoner) so the
    handshake parsing rules - notification discard, error surfacing,
    timeout bound - cannot drift between the two callers.

    Note: a "result" here is a JSON-RPC response with a matching id.
    This function explicitly does NOT dispatch to any text-delta or
    server-request hook; the conversational backend re-adds hook
    dispatch on top inside `_send_locked`, and the one-shot reasoner
    drives its own message-pump loop using this function and the
    `write_rpc` primitive for the handshake then switches to a
    per-message dispatch loop for `session/prompt`.
    """
    assert proc is not None and proc.stdout is not None
    while True:
        try:
            line = await asyncio.wait_for(
                proc.stdout.readline(),
                timeout=timeout_seconds,
            )
        except TimeoutError as exc:
            raise TimeoutError(
                f"{backend_label} ACP handshake timed out waiting for response id={expected_id}"
            ) from exc

        if not line:
            raise RuntimeError(f"{backend_label} process exited during handshake")

        try:
            msg = json.loads(line.decode())
        except json.JSONDecodeError:
            # Non-JSON output during startup (e.g., progress bars).
            continue

        # Check for matching response.
        if msg.get("id") == expected_id:
            if "error" in msg:
                err = msg["error"]
                raise RuntimeError(
                    f"{backend_label} ACP error (id={expected_id}): {err.get('message', 'unknown error')}"
                )
            return msg.get("result", {})

        # Discard notifications (session/update during startup).


# Quiet window for draining text chunks that land on stdout AFTER the
# session/prompt response. OpenCode resolves the prompt request when
# the session goes idle, but its event forwarding is asynchronous: the
# final agent_message_chunk notification(s) can flush milliseconds
# after the response (observed lag ~100ms on opencode 1.15.11). The
# window restarts on every received line, so it bounds the wait for
# the NEXT line, not the total drain time; 0.5s gives roughly 5x
# margin over the observed lag while capping the added per-call
# latency when nothing trails the response.
COMPLETION_DRAIN_WINDOW_S: float = 0.5


async def drain_late_text(
    *,
    proc: asyncio.subprocess.Process | None,
    accumulated: str,
    extract_delta: Callable[[dict], str | None],
    combine: Callable[[str, str], str],
    window_s: float | None = None,
) -> str:
    """
    Drain trailing text-chunk notifications after a prompt response
    and return the updated accumulated text.

    The ACP read loops treat the matching-id response to
    session/prompt as end-of-turn, but the response can beat the
    turn's final text chunk(s) onto stdout (see
    COMPLETION_DRAIN_WINDOW_S above for the mechanism). Stopping at
    the response therefore truncates the tail of the answer: fatal
    for one-shot JSON consumers, whose whole response becomes
    unparseable, and wrong for the conversational loop, where the
    unread chunk stays buffered in the pipe and surfaces at the START
    of the next turn's reply.

    Reads lines until the quiet window passes with no output or the
    pipe reaches EOF, feeding each notification through the caller's
    `extract_delta` / `combine` hooks so opencode's sentence-boundary
    whitespace join applies to drained chunks exactly as it does to
    in-turn chunks. Non-text notifications and non-JSON lines are
    skipped. Server-initiated requests are not answered here: the
    turn is already complete, so no permission request can be
    pending, and anything else with an id is a response to a request
    this client never sent.

    EOF ends the drain rather than raising: by this point the turn
    succeeded and the accumulated text is the answer. The one-shot
    caller kills the subprocess right after this returns anyway, and
    the conversational caller's next send() handles respawn.

    `window_s=None` resolves COMPLETION_DRAIN_WINDOW_S at call time
    so tests can patch the module constant and cover both call sites
    with one knob.
    """
    if window_s is None:
        window_s = COMPLETION_DRAIN_WINDOW_S
    assert proc is not None and proc.stdout is not None
    while True:
        try:
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=window_s)
        except TimeoutError:
            return accumulated
        if not line:
            return accumulated
        try:
            msg = json.loads(line.decode())
        except json.JSONDecodeError:
            continue
        if "method" in msg and "id" not in msg:
            text = extract_delta(msg)
            if text:
                accumulated = combine(accumulated, text)


def convert_image_block(block: dict) -> dict | None:
    """
    Convert an Anthropic-style base64 image block to the ACP shape.

    The bot layer builds image content in the Anthropic form
    (`{"type": "image", "source": {"type": "base64", "media_type":
    ..., "data": ...}}`) because the claude backend forwards content
    blocks verbatim. ACP defines its own image content block
    (`{"type": "image", "mimeType": ..., "data": ...}` with base64
    data), so the ACP path translates rather than asking the bot
    layer to grow per-backend block shapes.

    Returns None when the block is not the expected Anthropic base64
    shape (non-dict source, a non-base64 source type, or missing
    fields); the caller drops such blocks the same way it drops
    unsupported block types.
    """
    source = block.get("source")
    if not isinstance(source, dict) or source.get("type") != "base64":
        return None
    media_type = source.get("media_type")
    data = source.get("data")
    if not isinstance(media_type, str) or not isinstance(data, str):
        return None
    return {"type": "image", "mimeType": media_type, "data": data}


# ── Shared cross-user subprocess helpers ─────────────────────────


# Timeout (seconds) for the cross-user kill subprocess on the wrap
# path. A hung kill must not stall the caller's own timeout cleanup
# (a one-shot reasoner's typed-error path, or the conversational
# backend's _kill/shutdown chain), so the cap is intentionally short
# relative to the timeouts those callers run under.
_CROSS_USER_KILL_TIMEOUT_S: float = 5.0


def _stderr_is_esrch(stderr: bytes | None) -> bool:
    """
    True iff `stderr` carries the POSIX ESRCH diagnostic from
    `/bin/kill` for a no-such-process condition.

    Discriminates the benign race (the target tree already exited
    between the caller's decision to kill and the signal delivery;
    kill returns rc=1 with "No such process") from real failure
    modes (sudoers misconfiguration, signal permission denied, kill
    binary missing). The benign case logs at DEBUG; everything else
    keeps the WARNING.

    Substring match because BSD `/bin/kill` on macOS and util-linux
    `kill` on Linux both emit `kill: <pid>: No such process`; the
    prefix varies slightly but the "No such process" portion is the
    stable POSIX `strerror(ESRCH)` text, fixed by libc and not
    localized for these binaries on any production platform.

    Returns False on `None` or empty bytes so a missing-stderr
    rc!=0 (itself unusual and probably a real failure) keeps the
    WARNING.
    """
    if not stderr:
        return False
    return b"No such process" in stderr


def _log_cross_user_kill_result(
    *,
    rc: int,
    stderr: bytes,
    target_user: str,
    pgid: int,
    purpose: str,
    backend: str,
) -> None:
    """
    Log the outcome of a cross-user kill subprocess.

    Shared by the async and sync kill variants so the rc/ESRCH
    classification cannot drift between them: rc=0 is silent
    success, ESRCH is the benign already-gone race at DEBUG, any
    other non-zero rc is a WARNING an operator needs to see.
    """
    if rc == 0:
        return
    if _stderr_is_esrch(stderr):
        log.debug(
            "cross-user kill: process group already gone (purpose=%s backend=%s target=%s pgid=%d); benign race",
            purpose,
            backend,
            target_user,
            pgid,
        )
        return
    log.warning(
        "cross-user kill returned rc=%d (purpose=%s backend=%s target=%s pgid=%d stderr=%r)",
        rc,
        purpose,
        backend,
        target_user,
        pgid,
        stderr[:200],
    )


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
    `install._generate_sudoers` already emits). The leading `-` on
    the PID arg makes kill's target a process group; sending to the
    negative group id signals every process the target user has
    permission to signal in that group, covering descendants at
    every tree depth without per-PID discovery.

    Shared by the one-shot reasoners (timeout escalation) and the
    conversational `AcpBackend` kill paths so the cross-user kill
    semantics cannot drift between them. Bounded by
    `_CROSS_USER_KILL_TIMEOUT_S` so a hung sudo or missing
    `/bin/kill` does not stall the caller's own cleanup. Failures
    (non-ESRCH rc, timeout, FileNotFoundError on sudo itself) are
    logged and swallowed; the caller still reaps the wrapper
    afterward, accepting the orphan risk for that deployment edge
    case.
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
    _log_cross_user_kill_result(
        rc=rc,
        stderr=stderr,
        target_user=target_user,
        pgid=pgid,
        purpose=purpose,
        backend=backend,
    )


def _kill_target_user_tree_sync(
    *,
    target_user: str,
    pgid: int,
    purpose: str,
    backend: str,
) -> None:
    """
    Synchronous companion to `_kill_target_user_tree` for sync-only
    callers.

    `AgentBackend.force_kill` is a synchronous method (the pool's
    last-resort path after `shutdown` failed or timed out), so it
    cannot await the canonical async helper; the claude and codex
    backends keep a sync signal path for exactly this reason. Same
    argv, timeout bound, and rc/ESRCH log classification as the
    async variant. Blocking the loop for up to
    `_CROSS_USER_KILL_TIMEOUT_S` is accepted on this path: it only
    runs when a graceful shutdown already failed and the
    alternative is an orphaned target-user process tree.
    """
    cmd = ["sudo", "-n", "-u", target_user, "/bin/kill", "-KILL", f"-{pgid}"]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=_CROSS_USER_KILL_TIMEOUT_S,
            check=False,
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
    except subprocess.TimeoutExpired:
        log.warning(
            "cross-user kill timed out (purpose=%s backend=%s target=%s pgid=%d)",
            purpose,
            backend,
            target_user,
            pgid,
        )
        return
    _log_cross_user_kill_result(
        rc=result.returncode,
        stderr=result.stderr,
        target_user=target_user,
        pgid=pgid,
        purpose=purpose,
        backend=backend,
    )


# ── ACP base backend ──────────────────────────────────────────────


class AcpBackend(AgentBackend):
    """
    AgentBackend implementation for the ACP JSON-RPC 2.0 wire shape.

    Manages a persistent subprocess lifecycle: two-step handshake
    (initialize + session/new), prompt dispatch via session/prompt,
    streaming response accumulation from session/update notifications,
    and kill / restart on demand.

    All message sends are serialized via an internal asyncio lock so
    concurrent callers queue rather than interleave. Subclasses supply
    the harness-specific details (argv, env vars, model mapping,
    streaming notification shapes) through the hook surface listed in
    the module docstring.
    """

    # Concrete subclasses MUST override both. The empty defaults are
    # inherited from AgentBackend (backend_name) and declared here
    # (backend_label) so the ABC stays importable for type stubs and
    # tests that build minimal fakes.
    backend_label: str = ""

    def __init__(
        self,
        *,
        model: str = "sonnet",
        workspace: Path = Path("home"),
        home_workspace: Path | None = None,
        webhook_port: int = 8080,
        webhook_secret: str = "",
        timeout_seconds: int = 120,
        services_info: list[dict] | None = None,
        workspace_config: WorkspaceConfig | None = None,
        provider: str = "",
        # Operator-intent flag for the memory subsystem (Config.
        # memory_enabled). Drives the [Memory subsystem: ...] marker
        # emission and gates MEMORY.md inject in build_session_context.
        # Default False so direct test instantiations need not plumb
        # the kwarg; production callers (pool.py) always pass an
        # explicit value.
        memory_enabled: bool = False,
        # Optional OS user to run the ACP subprocess as, via
        # `sudo -H -u <user>`. None = run as the bot's process user.
        # Same per-user isolation contract as claude_user / codex_user
        # on the sibling backends; resolved through resolve_claude_user
        # at spawn time so a value naming the bot's own user collapses
        # to the direct-spawn path (self-sudo skip).
        os_user: str | None = None,
    ):
        # ABC-required attributes (pool.py reads/writes these)
        self.model = model
        self.workspace = workspace
        self.home_workspace = home_workspace or workspace
        self.timeout_seconds = timeout_seconds
        self.workspace_config = workspace_config
        self.provider = provider
        self.memory_enabled = memory_enabled
        self.os_user = os_user

        # API context for session injection (passed to build_session_context).
        self._api_context = ApiContext(
            webhook_port=webhook_port,
            webhook_secret=webhook_secret,
            services_info=services_info or [],
        )

        # Global defaults, preserved so we can restore them when
        # switching away from a configured workspace.
        self._default_model = model
        self._default_timeout = timeout_seconds

        # Apply per-workspace overrides (if configured). Model validated
        # against the active backend's surface via apply_workspace_model;
        # the helper takes the backend identifier as a parameter so
        # `self.backend_name` (set as a class attribute by the concrete
        # subclass) drives the same validation goose used to hardcode
        # to the "goose" literal.
        if workspace_config:
            self.model = apply_workspace_model(workspace_config, self.backend_name, self.provider, self.model)
            if workspace_config.timeout is not None:
                self.timeout_seconds = workspace_config.timeout

        # Subprocess and session state.
        self._proc: asyncio.subprocess.Process | None = None
        self._session_id: str | None = None
        # Cross-user spawn state, set per incarnation in
        # _ensure_started. `_effective_os_user` is the resolved sudo
        # target (None on the direct path); `_pgid` is the wrapper's
        # process group id, recorded only on the wrap path so the
        # kill paths know whether a sudo escalation is needed and
        # which group to target. _kill nulls both so a recycled
        # instance cannot carry stale routing into a fresh spawn.
        self._effective_os_user: str | None = None
        self._pgid: int | None = None
        # Whether the agent accepts image content blocks on
        # session/prompt, read from `promptCapabilities.image` in the
        # initialize result. False until a handshake completes so a
        # prompt can never race ahead of the capability answer; each
        # handshake overwrites it, so a restart re-reads the agent's
        # current answer.
        self._supports_image_input: bool = False
        # Monotonically increasing JSON-RPC request ID. Reset to 1 on
        # each subprocess start; the request IDs are scoped to one
        # subprocess incarnation so the read loop's id-matching is
        # unambiguous across restarts.
        self._next_id: int = 1
        self._lock = asyncio.Lock()  # Serializes all message sends
        # True until the first send() finishes; first-message context
        # (CLAUDE.md / PREFERENCES.md / session_context) is injected
        # exactly once per session.
        self._fresh_session: bool = True
        self._stderr_task: asyncio.Task | None = None

    # ── Hooks for concrete adapters ──────────────────────────────────

    def build_argv(self) -> list[str]:
        """
        Return the subprocess argv vector.

        Hook may consult self.model when the harness accepts a `--model`
        flag on argv (e.g., OpenCode's `opencode acp --model <full-id>`).
        Goose injects model via GOOSE_MODEL env instead and so returns
        a static vector.
        """
        raise NotImplementedError

    def build_env(self, base_env: dict[str, str]) -> dict[str, str]:
        """
        Layer backend-specific env vars onto base_env and return it.

        Goose adds GOOSE_MODEL (and conditionally GOOSE_PROVIDER).
        OpenCode may add OPENCODE_CONFIG_CONTENT or OPENCODE_CONFIG.
        The shared layer applies workspace env_file / inline env and
        the webhook secret AFTER this hook so workspace overrides can
        still shadow backend-specific defaults; the webhook secret is
        applied last so workspace env cannot override it.

        Hook receives a mutable dict copy of the inherited environment;
        it may mutate in place or return a different dict.
        """
        raise NotImplementedError

    def preserved_env_vars(self) -> tuple[str, ...]:
        """
        Return the env var names the cross-user sudo wrap forwards
        through sudo's env_reset (the `--preserve-env=<csv>` clause).

        Only consulted on the wrap path (`os_user` resolved to a
        non-bot user); the direct spawn hands the full env built in
        `_ensure_started` to the subprocess unfiltered. The per-user
        sudoers entries generated by `install._generate_sudoers`
        carry the `SETENV:` tag that authorizes the passthrough.

        Default mirrors claude.py's wrap: KAI_WEBHOOK_SECRET (the
        per-session token the agent needs to call back into Kai's
        webhook API) and TMPDIR (the per-os-user temp anchor set in
        `_ensure_started`; without preservation env_reset strips the
        anchor and the agent falls back to the shared /tmp).
        Concrete adapters whose harness reads configuration or auth
        from the environment must override and EXTEND this list -
        sudo strips everything not named here, so an adapter that
        delivers model selection via env (GooseBackend) loses it
        silently otherwise.
        """
        return ("KAI_WEBHOOK_SECRET", "TMPDIR")

    def build_initialize_params(self) -> dict:
        """
        Return the params for the initialize JSON-RPC call.

        Default matches Goose's payload (protocolVersion=v1, clientInfo
        identifying Kai). Concrete adapters override only if the harness
        requires a different shape.
        """
        return {
            "protocolVersion": "v1",
            "clientInfo": {"name": "kai", "version": kai.__version__},
        }

    def build_session_new_params(self) -> dict:
        """
        Return the params for the session/new JSON-RPC call.

        Backend-specific because session creation can require harness
        details (Goose: `cwd` + empty `mcpServers` array; OpenCode may
        differ once the wire-shape smoke is recorded).
        """
        raise NotImplementedError

    def extract_session_id(self, result: dict) -> str:
        """
        Pull the session ID out of the session/new result payload.

        Default reads `sessionId` (the ACP standard). Override if the
        harness uses a different field name.
        """
        return result["sessionId"]

    def extract_text_delta(self, msg: dict) -> str | None:
        """
        Return user-visible assistant text from a session/update notification.

        Returns the text chunk when this notification carries assistant
        output that should reach the user; returns None for tool-call,
        agent-thought, or any other notification shape the harness emits
        but Kai should not surface.
        """
        raise NotImplementedError

    def is_completion(self, msg: dict, prompt_id: int) -> bool:
        """
        True iff `msg` is the final-success response for prompt_id.

        Default matches the ACP shape: a JSON-RPC object with the
        request's id and a `result` field. Override only if the harness
        diverges.
        """
        return msg.get("id") == prompt_id and "result" in msg

    def extract_error(self, msg: dict, prompt_id: int) -> str | None:
        """
        Return the error message string when `msg` is a JSON-RPC error
        response for prompt_id; otherwise None.

        Default matches the ACP shape: matching id + `error` field with
        a `message` string. Override if the harness emits errors with a
        different structure.
        """
        if msg.get("id") == prompt_id and "error" in msg:
            return msg["error"].get("message", "unknown ACP error")
        return None

    def combine_text_chunks(self, prev: str, new: str) -> str:
        """
        Combine two streamed text chunks during prompt-response
        accumulation. Default is verbatim concatenation (`prev + new`),
        which is right for Goose and any future ACP harness whose
        chunk boundaries already carry their own whitespace.
        OpenCode overrides this to inject a single space at
        sentence-boundary joins where the model dropped the space
        between chunks; see `OpenCodeBackend.combine_text_chunks`
        and `kai.opencode.concat_opencode_text` for the heuristic
        and rationale.

        The hook fires per chunk pair, not per accumulated string,
        so backends that need a more elaborate normalization can
        keep per-pair state in instance attributes if they need to.
        Today only the simple lookahead-at-last-char shape is used.
        """
        return prev + new

    def handle_server_request(self, msg: dict) -> dict | None:
        """
        Return a JSON-RPC `result` payload for a server-initiated request.

        Some ACP harnesses (OpenCode in particular) emit JSON-RPC
        requests TO the client mid-stream, expecting a matching-id
        response before the prompt can continue. The classic example is
        `session/request_permission` for a tool call: OpenCode waits for
        the client to pick an `optionId` before the tool runs. Goose
        never does this (auto-approves via `--with-builtin developer`),
        so the default returns None (the shared read loop logs and
        skips) and Goose's behavior is unchanged.

        Concrete adapters override to recognize specific server methods
        and return the response body. The shared layer wraps the
        returned dict as `{jsonrpc: "2.0", id: <request_id>, result: <dict>}`
        and writes it back on stdin. Returning None for an unrecognized
        request method is the safe default; the harness either retries,
        times out, or proceeds without the answer.
        """
        return None

    async def _send_server_response(self, request_id: int, result: dict) -> None:
        """
        Write a JSON-RPC response back to the subprocess for a
        server-initiated request.

        Distinct from `_write_rpc` (which sends a new request and
        consumes a fresh `_next_id`); this writes a response using the
        id the SERVER sent us. Helper kept out of the read loop so the
        ordering and error handling stay clear.
        """
        assert self._proc is not None and self._proc.stdin is not None
        payload = {"jsonrpc": "2.0", "id": request_id, "result": result}
        line = (json.dumps(payload) + "\n").encode()
        self._proc.stdin.write(line)
        await self._proc.stdin.drain()

    # ── Lifecycle properties ──────────────────────────────────────────

    @property
    def is_alive(self) -> bool:
        """True if the ACP subprocess is running and hasn't exited."""
        return self._proc is not None and self._proc.returncode is None

    @property
    def session_id(self) -> str | None:
        """The current ACP session ID, or None if no session is active."""
        return self._session_id

    # ── Process lifecycle ──────────────────────────────────────────

    async def _ensure_started(self) -> None:
        """
        Start the ACP subprocess if not already running.

        Spawns the subprocess via build_argv() with the env produced by
        build_env() (plus workspace overrides and the webhook secret),
        then runs the two-step handshake:
        1. initialize - establishes protocol version and capabilities
        2. session/new - creates a session with backend-specific params

        When `os_user` resolves to a non-bot user, the argv is wrapped
        in `sudo -H -u <target> --preserve-env=<csv> --` so the agent
        runs as that user (per-user OS isolation, same contract as the
        claude and codex backends). `-H` rewrites HOME so the agent
        reads its config and auth state under the target user's home;
        the preserve list comes from the preserved_env_vars() hook.

        The process persists across prompts. Restarts (via /new or a
        workspace switch) re-run the handshake with a fresh session.
        """
        if self.is_alive:
            return

        # Per-user OS routing. resolve_claude_user (named claude-
        # historically; its body is backend-agnostic) returns None
        # when os_user is unset OR matches the bot's own user, so the
        # self-sudo-skip path is byte-identical to a no-os_user spawn.
        effective_os_user = resolve_claude_user(self.os_user)

        # Build the subprocess environment. Layering order:
        # 1. Base environment (inherited from parent process)
        # 2. Backend-specific keys (via self.build_env)
        # 3. Per-workspace env_file values
        # 4. Per-workspace inline env values (override env_file)
        # 5. Webhook secret (workspace env can't override it)
        # 6. Per-os-user TMPDIR anchor (cross-user mode only; LAST so
        #    workspace env cannot point one user's temp writes at
        #    another's). The per-user dirs under <DATA_DIR>/tmp/ are
        #    created and chowned by install.py; TMPDIR survives
        #    sudo's env_reset via the preserved_env_vars() default.
        env = self.build_env(os.environ.copy())
        if self.workspace_config:
            if self.workspace_config.env_file:
                env.update(parse_env_file(self.workspace_config.env_file))
            if self.workspace_config.env:
                env.update(self.workspace_config.env)
        if self._api_context.webhook_secret:
            env["KAI_WEBHOOK_SECRET"] = self._api_context.webhook_secret
        if effective_os_user:
            env["TMPDIR"] = str(DATA_DIR / "tmp" / effective_os_user)

        argv = self.build_argv()
        if effective_os_user:
            # The SETENV: tag on the per-os_user sudoers rule
            # authorizes --preserve-env; the rule pins the absolute
            # agent binary path, so sudo's PATH resolution of the
            # argv head must land on the same file the rule names.
            preserve = ",".join(self.preserved_env_vars())
            argv = [
                "sudo",
                "-H",
                "-u",
                effective_os_user,
                f"--preserve-env={preserve}",
                "--",
                *argv,
            ]

        log.info(
            "Starting persistent %s ACP process (model=%s, user=%s)",
            self.backend_label,
            self.model,
            effective_os_user or "(same as bot)",
        )

        self._proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.workspace),
            env=env,
            limit=1024 * 1024,  # 1MB per-line buffer
            # Cross-user mode: new session group so the sudo wrapper
            # is the session leader (PGID == PID) and the kill paths
            # can SIGKILL the whole target-user tree by negative
            # group id. Direct mode keeps default semantics so a
            # plain kill on _proc reaches the agent alone.
            start_new_session=bool(effective_os_user),
        )
        # PGID == PID for session leaders. Recorded now because
        # os.getpgid() fails once the wrapper exits, while the group
        # kill works as long as any member survives (the orphaned
        # agent is exactly the member that survives the wrapper).
        self._pgid = self._proc.pid if effective_os_user else None
        self._effective_os_user = effective_os_user
        self._stderr_task = asyncio.create_task(self._drain_stderr())

        # Step 1: initialize - establish protocol version and read the
        # agent's capability answer. `promptCapabilities.image` gates
        # whether image content blocks are forwarded on session/prompt
        # or dropped with a user-visible notice. The chained
        # isinstance guards treat any missing or malformed capability
        # structure as "no image support": dropping an image a capable
        # agent could have read is a degraded reply, but sending an
        # image to an agent that never advertised support is a
        # protocol violation with harness-defined behavior.
        self._next_id = 1
        await self._write_rpc("initialize", self.build_initialize_params())
        init_result = await self._read_result(expected_id=1)
        agent_caps = init_result.get("agentCapabilities")
        prompt_caps = agent_caps.get("promptCapabilities") if isinstance(agent_caps, dict) else None
        self._supports_image_input = bool(prompt_caps.get("image")) if isinstance(prompt_caps, dict) else False

        # Step 2: session/new - create a session.
        await self._write_rpc("session/new", self.build_session_new_params())
        result = await self._read_result(expected_id=2)
        self._session_id = self.extract_session_id(result)
        self._fresh_session = True

    async def _drain_stderr(self) -> None:
        """
        Continuously read and discard stderr from the ACP process.

        Without this, the stderr pipe buffer fills up and the process
        deadlocks. Lines are logged at DEBUG level for routine
        diagnostics; lines that match an upstream-error indicator
        (`permission`, `rejected`, `invalid`, `schema`) are surfaced
        at WARNING so operators see ACP-server-side rejections without
        having to enable DEBUG. The backend label prefixes the log
        line so a multi-backend deployment can tell streams apart.

        Without the selective bump, an ACP server like OpenCode can
        reject a client response as schema-invalid and the diagnostic
        is silently swallowed at DEBUG level, leaving an operator
        without any signal that the upstream complained.
        """
        while self._proc and self._proc.stderr:
            try:
                line = await self._proc.stderr.readline()
                if not line:
                    break
                text = line.decode().strip()
                if not text:
                    continue
                # Case-insensitive substring check is enough; the
                # tokens are distinctive within ACP-server diagnostic
                # output, and a quoted-string false positive would only
                # mean one extra WARNING line per occurrence, which is
                # an acceptable cost for the load-bearing real signal.
                lowered = text.lower()
                if any(token in lowered for token in _STDERR_WARNING_TOKENS):
                    log.warning("%s stderr: %s", self.backend_label, text[:200])
                else:
                    log.debug("%s stderr: %s", self.backend_label, text[:200])
            except Exception:
                log.warning("Unexpected error in %s stderr drain", self.backend_label, exc_info=True)
                break

    # ── JSON-RPC helpers ───────────────────────────────────────────

    async def _write_rpc(self, method: str, params: dict) -> None:
        """
        Write a JSON-RPC 2.0 request to the subprocess stdin.

        Thin wrapper over the module-level `write_rpc` free function so
        the one-shot reasoner (`kai.oneshot.OpenCodeOneShotReasoner`)
        and this persistent backend share the same wire-shape
        primitive. Mutating `self._next_id` and forwarding the rest of
        the instance state preserves the conversational backend's
        existing semantics; the read loop's id-matching depends on the
        counter being incremented exactly once per request.
        """
        self._next_id = await write_rpc(
            proc=self._proc,
            next_id=self._next_id,
            method=method,
            params=params,
        )

    async def _read_result(self, expected_id: int) -> dict:
        """
        Read stdout lines until a JSON-RPC result with the expected id appears.

        Thin wrapper over the module-level `read_result` free function.
        Shared with the one-shot reasoner so handshake parsing rules
        (discard notifications, surface JSON-RPC errors, time-bound on
        `timeout_seconds`) do not drift between the two callers.
        """
        return await read_result(
            proc=self._proc,
            expected_id=expected_id,
            timeout_seconds=self.timeout_seconds,
            backend_label=self.backend_label,
        )

    # ── Sending prompts ────────────────────────────────────────────

    async def send(self, prompt: str | list, chat_id: int | None = None) -> AsyncIterator[StreamEvent]:
        """
        Send a message to the ACP subprocess and yield streaming events.

        Serialized via an internal lock so concurrent callers queue
        rather than interleave. Context injection (identity, memory,
        history, API docs) is prepended on the first message of each
        session.

        Args:
            prompt: Either a text string or a list of content blocks.
            chat_id: Optional Telegram chat ID for history scoping
                and API routing.

        Yields:
            StreamEvent objects with accumulated text. The final event
            has done=True and includes the complete AgentResponse.
        """
        async with self._lock:
            async for event in self._send_locked(prompt, chat_id):
                yield event

    async def _send_locked(self, prompt: str | list, chat_id: int | None = None) -> AsyncIterator[StreamEvent]:
        """
        Core send logic (must be called while holding self._lock).

        Handles context injection, prompt coercion to ACP format,
        JSON-RPC session/prompt dispatch, and streaming response
        accumulation from session/update notifications.
        """
        try:
            await self._ensure_started()
        except (OSError, RuntimeError, TimeoutError) as exc:
            yield StreamEvent(
                text_so_far="",
                done=True,
                response=AgentResponse(success=False, text="", error=f"{self.backend_label} startup failed: {exc}"),
            )
            return

        # Per-turn prompt context. Build the fresh-session bootstrap
        # (MEMORY.md / PREFERENCES.md seed + session_context block) and
        # the foreign-workspace reminder LOCALLY, then hand both to
        # `assemble_turn_context` so the shared helper owns the
        # ordering invariant (USER_MESSAGE_MARKER closest to user text;
        # session_context, semantic memory, workspace reminder stacked
        # above).
        #
        # The ensure_user_memory / ensure_user_preferences calls below
        # run exactly once per session because _fresh_session flips to
        # False unconditionally. A transient OSError inside either
        # helper is logged as a warning but not retried on subsequent
        # messages of the same session. Failure modes are persistent
        # (permissions, missing parent dir), not transient, so the
        # one-shot behavior is acceptable.
        session_ctx = ""
        if self._fresh_session:
            self._fresh_session = False
            ensure_user_memory(chat_id, DATA_DIR)
            ensure_user_preferences(chat_id, DATA_DIR)
            session_ctx = build_session_context(
                workspace=self.workspace,
                home_workspace=self.home_workspace,
                api=self._api_context,
                workspace_config=self.workspace_config,
                chat_id=chat_id,
                data_dir=DATA_DIR,
                memory_enabled=self.memory_enabled,
            )

        reminder = build_foreign_workspace_reminder(self.workspace, self.home_workspace) or ""

        # Normalize user blocks to the ACP content shape before
        # per-turn assembly. Text blocks pass through; image blocks
        # are converted to ACP's image shape when the agent advertised
        # `promptCapabilities.image` on this handshake, and dropped
        # otherwise (capability absent, or a block that is not the
        # Anthropic base64 shape `convert_image_block` expects). Every
        # dropped image is counted so the reply can carry a
        # user-visible notice; a log-only warning leaves the user
        # believing the model saw an image it never received.
        # Normalization must run BEFORE assemble_turn_context so the
        # marker it prepends labels a real user region. Running it
        # after the helper would leave injected text layers
        # (session_context, reminder, memory) above the marker and
        # nothing below it.
        had_user_text = isinstance(prompt, str)
        dropped_images = 0
        if isinstance(prompt, list):
            acp_blocks: list[dict] = []
            for block in prompt:
                block_type = block.get("type")
                if block_type == "text":
                    acp_blocks.append({"type": "text", "text": block["text"]})
                    continue
                if block_type == "image" and self._supports_image_input:
                    converted = convert_image_block(block)
                    if converted is not None:
                        acp_blocks.append(converted)
                        continue
                if block_type == "image":
                    dropped_images += 1
                log.warning(
                    "%s: dropping content block type=%s (supports_image_input=%s)",
                    self.backend_label,
                    block_type,
                    self._supports_image_input,
                )
            had_user_text = any(b.get("type") == "text" for b in acp_blocks)
            # ACP requires a non-empty prompt array. Input whose every
            # block was dropped becomes a single placeholder so the
            # marker has a user region to label.
            prompt = acp_blocks or [{"type": "text", "text": "(empty prompt)"}]

        # `chat_id=None` to the helper suppresses semantic recall for
        # this turn. The placeholder "(empty prompt)" is backend-
        # synthetic and must not become a memory search query; only
        # real user text drives recall. Session context still uses the
        # real chat_id - it was built above this point.
        recall_chat_id = chat_id if had_user_text else None
        prompt = await assemble_turn_context(
            prompt,
            chat_id=recall_chat_id,
            session_context=session_ctx,
            workspace_reminder=reminder,
            workspace=self.workspace,
            backend_name=self.backend_name,
            job_type="interactive",
        )

        # Coerce to the ACP content-block shape. `prompt` is either a
        # str (from a str input; the helper preserves the input type
        # family) or a list already normalized to ACP blocks above
        # (text plus any forwarded images).
        acp_prompt: list[dict]
        if isinstance(prompt, str):
            acp_prompt = [{"type": "text", "text": prompt}]
        else:
            acp_prompt = prompt

        # Send the session/prompt request.
        assert self._proc is not None
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None

        try:
            prompt_id = self._next_id
            await self._write_rpc(
                "session/prompt",
                {
                    "sessionId": self._session_id,
                    "prompt": acp_prompt,
                },
            )
        except OSError as e:
            log.error("Failed to write to %s process: %s", self.backend_label, e)
            await self._kill()
            yield StreamEvent(
                text_so_far="",
                done=True,
                response=AgentResponse(
                    success=False,
                    text="",
                    error=f"{self.backend_label} process died, restarting on next message",
                ),
            )
            return

        # Stream response: read stdout lines, accumulate text via the
        # extract_text_delta hook, and yield StreamEvents.
        #
        # When images were dropped above, the display text is seeded
        # with a notice so the user learns the model never saw them;
        # the drop is otherwise invisible (the reply reads as a normal
        # answer to the caption text). Seeding `accumulated` puts the
        # notice in every StreamEvent and in the final AgentResponse
        # without touching the response protocol. The trailing
        # newlines end the seed at a line break, so chunk-join
        # heuristics (see combine_text_chunks) never fire against it.
        # `got_model_text` exists because the seed makes `accumulated`
        # non-empty before the agent says anything; the EOF branch
        # below must judge "did the model produce output" from this
        # flag, not from `accumulated` truthiness, or a process that
        # dies silently after an image drop would surface as a
        # successful notice-only reply.
        accumulated = ""
        got_model_text = False
        if dropped_images:
            noun = "image" if dropped_images == 1 else "images"
            accumulated = (
                f"[Note: {dropped_images} attached {noun} could not be passed to "
                f"{self.backend_label}; this reply is based on the message text only.]\n\n"
            )
        last_activity = time.monotonic()
        max_idle_seconds = self.timeout_seconds * 5

        try:
            while True:
                # Check idle timeout before each readline.
                idle = time.monotonic() - last_activity
                if idle > max_idle_seconds:
                    log.error(
                        "%s idle timeout (%.0fs with no output, limit %ds)",
                        self.backend_label,
                        idle,
                        max_idle_seconds,
                    )
                    await self._kill()
                    yield StreamEvent(
                        text_so_far=accumulated,
                        done=True,
                        response=AgentResponse(
                            success=False,
                            text=accumulated,
                            error=f"{self.backend_label} interaction timed out (no output)",
                        ),
                    )
                    return

                try:
                    line = await asyncio.wait_for(
                        self._proc.stdout.readline(),
                        timeout=self.timeout_seconds * 3,
                    )
                except TimeoutError:
                    log.error("%s response timed out", self.backend_label)
                    await self._kill()
                    yield StreamEvent(
                        text_so_far=accumulated,
                        done=True,
                        response=AgentResponse(
                            success=False,
                            text=accumulated,
                            error=f"{self.backend_label} timed out",
                        ),
                    )
                    return

                # Reset idle timer on non-empty output.
                if line:
                    last_activity = time.monotonic()
                else:
                    # EOF - process died. Success iff the model said
                    # anything; the notice seed alone does not count
                    # (see `got_model_text` above).
                    log.error("%s process EOF", self.backend_label)
                    await self._kill()
                    yield StreamEvent(
                        text_so_far=accumulated,
                        done=True,
                        response=AgentResponse(
                            success=got_model_text,
                            text=accumulated,
                            error=None if got_model_text else f"{self.backend_label} process ended unexpectedly",
                        ),
                    )
                    return

                try:
                    msg = json.loads(line.decode())
                except json.JSONDecodeError:
                    log.debug("Skipping non-JSON line: %s", line[:200])
                    continue

                # Streaming notifications (no id field) - extract user-
                # visible text via the hook and accumulate; skip every
                # other notification shape (tool calls, thoughts, etc.).
                # `combine_text_chunks` is the seam where OpenCode
                # injects sentence-boundary whitespace; Goose's default
                # is verbatim `prev + new`.
                if "method" in msg and "id" not in msg:
                    text = self.extract_text_delta(msg)
                    if text:
                        accumulated = self.combine_text_chunks(accumulated, text)
                        got_model_text = True
                        yield StreamEvent(text_so_far=accumulated)
                    continue

                # Server-initiated JSON-RPC request (has BOTH method AND
                # id). OpenCode emits these for tool-permission decisions
                # and waits on a matching-id response before continuing.
                # The shape-keyed branch keeps the no-backend_name-in-
                # shared-loop rule intact: the hook does the per-backend
                # work. Goose never reaches this branch in practice
                # because its `--with-builtin developer` auto-approves.
                if "method" in msg and "id" in msg:
                    server_id = msg["id"]
                    result = self.handle_server_request(msg)
                    if result is not None:
                        try:
                            await self._send_server_response(server_id, result)
                        except OSError as e:
                            log.error(
                                "Failed to write server-request response to %s: %s",
                                self.backend_label,
                                e,
                            )
                    else:
                        log.debug(
                            "%s emitted server request method=%s id=%s; hook returned None, skipping",
                            self.backend_label,
                            msg.get("method"),
                            server_id,
                        )
                    continue

                # Final result for our prompt (has matching id).
                if self.is_completion(msg, prompt_id):
                    # The response can beat the turn's final text
                    # chunk(s) onto stdout; without the drain those
                    # chunks stay buffered in the pipe and surface at
                    # the start of the NEXT turn's reply (the
                    # subprocess persists across turns). See
                    # drain_late_text for the mechanism.
                    accumulated = await drain_late_text(
                        proc=self._proc,
                        accumulated=accumulated,
                        extract_delta=self.extract_text_delta,
                        combine=self.combine_text_chunks,
                    )
                    response = AgentResponse(
                        success=True,
                        text=accumulated,
                        session_id=self._session_id,
                        duration_ms=0,
                    )
                    yield StreamEvent(
                        text_so_far=accumulated,
                        done=True,
                        response=response,
                    )
                    return

                # JSON-RPC error for our prompt.
                err = self.extract_error(msg, prompt_id)
                if err is not None:
                    yield StreamEvent(
                        text_so_far=accumulated,
                        done=True,
                        response=AgentResponse(
                            success=False,
                            text=accumulated,
                            error=err,
                        ),
                    )
                    return

        except Exception as e:
            log.exception("Unexpected error reading %s stream", self.backend_label)
            await self._kill()
            yield StreamEvent(
                text_so_far=accumulated,
                done=True,
                response=AgentResponse(
                    success=False,
                    text=accumulated,
                    error=str(e),
                ),
            )

    # ── Kill / restart / shutdown ──────────────────────────────────

    def force_kill(self) -> None:
        """
        Kill the subprocess immediately via SIGKILL.

        Safe to call without holding the lock. Called by /stop to abort
        an in-flight response. Does NOT null _proc - full cleanup
        happens in _kill() when the read loop detects EOF.

        Cross-user mode: _proc is the sudo wrapper, owned by the
        service user; the agent underneath is owned by the target
        user, and SIGKILL cannot be relayed by sudo. Killing only the
        wrapper would orphan the agent, so the target-user process
        group is sudo-escalated FIRST (sync variant; this method is
        the pool's synchronous last resort). _pgid is nulled after
        the escalation so the async kill paths that follow (read
        loop EOF -> _kill) do not re-kill an already-dead group.
        """
        if self._proc is not None:
            if self._effective_os_user is not None and self._pgid is not None:
                _kill_target_user_tree_sync(
                    target_user=self._effective_os_user,
                    pgid=self._pgid,
                    purpose="chat",
                    backend=self.backend_name,
                )
                self._pgid = None
            try:
                self._proc.kill()
            except OSError:
                pass
        if self._stderr_task:
            self._stderr_task.cancel()
            self._stderr_task = None

    async def _kill(self) -> None:
        """
        Kill the subprocess and clean up all process state.

        Sends SIGKILL, waits up to 5 seconds for exit, then nulls all
        process references. Idempotent. Cross-user mode escalates the
        target-user group kill through the canonical async helper
        BEFORE the wrapper is reaped, and nulls _pgid so the
        force_kill() call below does not repeat the escalation
        synchronously.
        """
        if self._proc:
            if self._effective_os_user is not None and self._pgid is not None:
                await _kill_target_user_tree(
                    target_user=self._effective_os_user,
                    pgid=self._pgid,
                    purpose="chat",
                    backend=self.backend_name,
                )
                self._pgid = None
            self.force_kill()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except TimeoutError:
                pass
            # force_kill() already cancelled _stderr_task and set it to
            # None, so no additional cleanup needed here.
            self._proc = None
            self._session_id = None
            self._pgid = None
            self._effective_os_user = None

    async def restart(self) -> None:
        """
        Kill the current process so the next send() starts fresh.

        Called by /new command and model switches. The next send()
        calls _ensure_started() which re-runs the full handshake
        with a new session.
        """
        await self._kill()
        self._fresh_session = True

    async def change_workspace(
        self,
        new_workspace: Path,
        workspace_config: WorkspaceConfig | None = None,
    ) -> None:
        """
        Switch to a new workspace directory and apply its config.

        Kills the current subprocess. The next send() restarts the
        ACP harness in the new directory with the new config applied.
        """
        await self._kill()
        self.workspace = new_workspace
        self.workspace_config = workspace_config

        # Revert to global defaults, then apply overrides. Prevents
        # stale values when switching from a fully-configured workspace
        # to a partially-configured one. Model validated against the
        # active backend's surface via self.backend_name.
        self.model = self._default_model
        self.timeout_seconds = self._default_timeout
        if workspace_config:
            self.model = apply_workspace_model(workspace_config, self.backend_name, self.provider, self.model)
            if workspace_config.timeout is not None:
                self.timeout_seconds = workspace_config.timeout
        # _fresh_session is set True by _ensure_started() on next send()

    async def shutdown(self) -> None:
        """
        Graceful shutdown. No save prompt - ACP has no equivalent.

        Sends SIGTERM first and waits up to 5 seconds for clean exit.
        Falls back to SIGKILL if the process doesn't terminate in time.

        Cross-user mode: the SIGTERM lands on the sudo wrapper, which
        relays catchable signals to the target-user agent, so the
        graceful path needs no escalation. The SIGKILL fallback DOES:
        SIGKILL cannot be relayed, so the target-user group is
        escalated through the canonical async helper before
        force_kill() reaps the wrapper (with _pgid nulled so
        force_kill skips its own sync escalation).
        """
        if self._proc:
            try:
                self._proc.terminate()
            except OSError:
                # Process already exited between the _proc check and
                # terminate(). Fall through to cleanup below.
                pass
            else:
                try:
                    await asyncio.wait_for(self._proc.wait(), timeout=5)
                except TimeoutError:
                    if self._effective_os_user is not None and self._pgid is not None:
                        await _kill_target_user_tree(
                            target_user=self._effective_os_user,
                            pgid=self._pgid,
                            purpose="chat",
                            backend=self.backend_name,
                        )
                        self._pgid = None
                    self.force_kill()
                    try:
                        await asyncio.wait_for(self._proc.wait(), timeout=5)
                    except TimeoutError:
                        log.warning("%s: process did not exit after SIGKILL", self.backend_label)
        if self._stderr_task:
            self._stderr_task.cancel()
            self._stderr_task = None
        self._proc = None
        self._session_id = None
        self._pgid = None
        self._effective_os_user = None
