"""
OpenAI Codex CLI subprocess backend.

Implements the AgentBackend ABC for the `codex app-server` JSON-RPC
protocol. Persistent subprocess per user, kept alive across messages,
restarted on /new or workspace switch. The wire framing is newline-
delimited JSON (NDJSON); each message is a JSON-RPC 2.0 envelope (the
`"jsonrpc":"2.0"` header may be omitted on the wire).

The codex app-server protocol uses its own thread/turn/item vocabulary
- NOT the session/* / agent_message_chunk shape goose uses. The
authoritative reference is `codex-rs/app-server/README.md` in the
openai/codex repo:

    Handshake (per connection):
      1. client `initialize` request (clientInfo + capabilities)
      2. server response (userAgent, codexHome, platformFamily, platformOs)
      3. client `initialized` notification (no id)
      4. client `thread/start` request -> thread object with thread.id

    Per-message:
      - client `turn/start` request (threadId, input[], optional model/cwd)
      - server streams notifications: item/started -> N x
        item/agentMessage/delta -> item/completed -> turn/completed.
      - text is accumulated from item/agentMessage/delta `delta` fields
        per itemId; the item/completed event carries the authoritative
        full text and overrides our delta accumulation when present.

Schema-drift posture: unknown notification methods and unrecognized
item types are logged at DEBUG and skipped rather than aborting the
stream. A future codex release that adds new item types must not
break the conversational stream. The codex CLI version is captured
in install metadata so a bump triggers a re-pin pass.

This module does NOT depend on OpenAI's Codex Python SDK; the wire
protocol is implemented directly to keep Kai's release schedule
decoupled from a vendor SDK's versioning.
"""

import asyncio
import base64
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import AsyncIterator
from pathlib import Path

import kai
from kai.acp import _signal_target_user_pid, _signal_target_user_pid_sync
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
    sanitize_agent_environment,
)
from kai.backend_registry import BackendRegistryError, backend_registry_is_authoritative, resolve_backend_command
from kai.config import DATA_DIR, WorkspaceConfig, parse_env_file, resolve_claude_user

log = logging.getLogger(__name__)


# Codex is openai-only in v1. Pool.py resolves self.model through the
# backend/provider model registry for codex-backed users, so the value
# is already in Codex's native model form ("gpt-5.6-sol", "gpt-5.5",
# "gpt-5.4-mini", etc.). No logical-name remapping is needed at this
# layer, unlike goose.py which translates "sonnet" / "opus" / "haiku"
# to claude-sonnet-4-6 etc for the Anthropic case.
# If a future codex version accepts a Kai-internal alias, add a map
# here and apply it before setting CODEX_MODEL on the subprocess env.


# Where per-turn image files for codex live. Each authenticated principal gets
# a non-listable child directory. The app-server's LocalImageUserInput takes a
# file path, not an inline payload, so the base64 image data the bot layer sends
# in content blocks must be materialized on disk for the duration of the turn.
# The directory sits under DATA_DIR/files because that tree is the established
# upload area; the workspace would also be readable but can be a git repository,
# where a transient binary invites accidental commits.
_TURN_IMAGE_DIR = DATA_DIR / "files" / "codex_turn_images"


def _resolve_default_codex_bin() -> str:
    """Resolve Codex to a stable absolute path when CODEX_BIN is unset.

    Cross-user launches go through sudoers, whose command matching is
    path-exact. Prefer common absolute install paths so runtime and the
    installer's sudoers fallback stay aligned after a users.yaml-only
    switch to the codex backend.
    """
    for candidate in ("/usr/local/bin/codex", "/opt/homebrew/bin/codex"):
        path = Path(candidate)
        if path.is_file() and os.access(path, os.X_OK):
            return candidate
    discovered = shutil.which("codex")
    return discovered or "codex"


def _grant_turn_image_read_access(path: Path, reader_user: str) -> None:
    """Grant exactly one cross-OS-user reader access to a private image.

    Kai owns the freshly created file and keeps its POSIX mode at ``0600``.
    A named ACL gives the isolated Codex OS user read-only access without
    making the image readable to every local account or granting Kai a broader
    sudo file-copy capability.

    macOS ships ACL support in ``/bin/chmod``. Linux uses ``setfacl``; if it is
    unavailable, the caller fails closed and drops the image with the existing
    user-visible notice instead of weakening permissions.
    """
    if sys.platform == "darwin":
        command = [
            "/bin/chmod",
            "+a",
            f"user:{reader_user} allow read,readattr,readextattr,readsecurity",
            str(path),
        ]
    elif sys.platform.startswith("linux"):
        setfacl = shutil.which("setfacl")
        if setfacl is None:
            raise OSError("setfacl is required for isolated Codex image input on Linux")
        command = [setfacl, "-m", f"u:{reader_user}:r--", str(path)]
    else:
        raise OSError(f"isolated Codex image input is unsupported on {sys.platform}")

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit status {result.returncode}"
        raise OSError(f"could not grant Codex image access to {reader_user}: {detail}")


def write_turn_image_file(
    block: dict,
    *,
    chat_id: int | None,
    reader_user: str | None,
) -> Path | None:
    """
    Materialize an Anthropic-style base64 image block as a file for
    codex's `localImage` input item.

    The bot layer builds image content in the Anthropic form
    (`{"type": "image", "source": {"type": "base64", "media_type":
    ..., "data": ...}}`) because the claude backend forwards content
    blocks verbatim. The codex app-server has no inline-payload image
    input; its `LocalImageUserInput` is `{"type": "localImage",
    "path": ...}`, so the payload is written to a temp file and the
    path rides the turn. The caller owns the file's lifetime and unlinks it
    when the turn ends.

    Files remain mode ``0600``. In cross-user mode, ``reader_user`` receives a
    read-only named ACL. The root and per-principal directories are mode
    ``0711``: the isolated process can traverse a path it was explicitly given,
    while other local users cannot list staged filenames. File permissions
    remain the confidentiality boundary even if a random filename is learned.

    Returns None when the block is not the expected Anthropic base64
    shape, the payload is not valid base64, or the write fails; the
    caller drops such blocks with a user-visible notice.
    """
    source = block.get("source")
    if not isinstance(source, dict) or source.get("type") != "base64":
        return None
    media_type = source.get("media_type")
    data = source.get("data")
    if not isinstance(media_type, str) or not isinstance(data, str):
        return None
    try:
        raw = base64.b64decode(data, validate=True)
    except ValueError:
        return None

    # Suffix from the media subtype so the file extension matches the
    # content (codex and the model both key on extensions for type
    # hints). A subtype that is not a plain token falls back to a
    # neutral suffix rather than letting caller-controlled text into
    # the filename.
    subtype = media_type.split("/", 1)[-1]
    suffix = f".{subtype}" if subtype.isalnum() else ".img"
    principal = "unscoped" if chat_id is None else str(chat_id)
    principal_dir = _TURN_IMAGE_DIR / principal
    path: Path | None = None
    try:
        _TURN_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        _TURN_IMAGE_DIR.chmod(0o711)
        principal_dir.mkdir(parents=True, exist_ok=True)
        principal_dir.chmod(0o711)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=suffix,
            prefix="turn-image-",
            dir=str(principal_dir),
            delete=False,
        ) as fh:
            path = Path(fh.name)
            fh.write(raw)
        path.chmod(0o600)
        if reader_user is not None:
            _grant_turn_image_read_access(path, reader_user)
    except OSError as e:
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        log.warning("CodexBackend: could not write turn image file: %s", e)
        return None
    return path


def _unlink_turn_images(paths: list[Path]) -> None:
    """Remove this turn's image files; missing files are fine."""
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError as e:
            log.warning("CodexBackend: could not remove turn image %s: %s", path, e)


# ── Codex CLI backend ─────────────────────────────────────────────


class CodexBackend(AgentBackend):
    """
    AgentBackend implementation for OpenAI Codex CLI's JSON-RPC protocol.

    Manages the lifecycle of a persistent `codex app-server` subprocess:
    starting with the initialize / initialized / thread/start handshake,
    sending prompts via turn/start, streaming responses from the
    item/* and turn/* notifications until turn/completed arrives, and
    killing/restarting on demand. self._session_id stores the codex
    `thread.id` for ABC consistency with the other backends.

    All message sends are serialized via an internal asyncio lock to
    prevent interleaving. Tool auto-approval is handled by codex's own
    config; this backend does not inject permission decisions.
    """

    # Stable identifier consumed by pool/bot dispatch and tagged into
    # the `memory.recall` line so log analysts can attribute recall
    # behavior to the caller backend.
    backend_name = "codex"

    def __init__(
        self,
        *,
        model: str = "gpt-5.4",
        workspace: Path = Path("home"),
        home_workspace: Path | None = None,
        webhook_port: int = 8080,
        webhook_secret: str = "",
        timeout_seconds: int = 120,
        services_info: list[dict] | None = None,
        workspace_config: WorkspaceConfig | None = None,
        provider: str = "openai",
        # Optional OS user to run codex as via `sudo -H -u <user>`. When
        # set, the codex subprocess runs as <codex_user> and reads its
        # auth token from ~<codex_user>/.codex/auth.json, NOT the kai
        # service user's home. This is the multi-user OAuth-isolation
        # lever: a kai install that runs as service user "kai" but has
        # users.yaml entries with os_user="daniel" / os_user="scott"
        # spawns a codex subprocess as the per-user os_user, picking
        # up that user's own codex login. Mirrors ClaudeCodeBackend's
        # claude_user kwarg. Default None means "run as the service
        # user" (single-user install or test fixture).
        codex_user: str | None = None,
        # Operator-intent flag for the memory subsystem (Config.
        # memory_enabled). Drives the [Memory subsystem: ...] marker
        # emission and gates MEMORY.md inject in build_session_context.
        # Default False so direct test instantiations need not plumb
        # the kwarg; production callers (pool.py) always pass an
        # explicit value. Codex installs support full semantic memory:
        # extraction dispatches per user through CodexOneShotReasoner,
        # the same as every backend in ONESHOT_REASONER_BACKENDS.
        memory_enabled: bool = False,
        # Hours before the subprocess is recycled (0 = no limit).
        # Same contract as the claude and ACP backends: prevents
        # unbounded memory growth in a long-lived agent subprocess.
        # The pool passes Config.agent_max_session_hours.
        max_session_hours: float = 0,
        # Reasoning effort forwarded as a `-c model_reasoning_effort=
        # <value>` config override on the spawn argv. Empty (default)
        # passes no override: codex falls back to the per-OS-user
        # ~/.codex/config.toml or the model default. Set-or-absent by
        # design (see Config.codex_effort_level); validated at config
        # load, so any non-empty value reaching this point is one of
        # the CLI's accepted tiers.
        codex_effort_level: str = "",
    ):
        # ABC-required attributes (pool.py reads/writes these)
        self.model = model
        self.workspace = workspace
        self.home_workspace = home_workspace or workspace
        self.timeout_seconds = timeout_seconds
        self.workspace_config = workspace_config
        self.provider = provider  # ABC-mandated; bot.py reads this
        self.codex_user = codex_user
        self.memory_enabled = memory_enabled
        self.codex_effort_level = codex_effort_level
        # Session-age recycling limit (ABC surface; checked by the
        # inherited _should_recycle at the top of _send_locked).
        self.max_session_hours = max_session_hours

        # API context for session injection (passed to build_session_context)
        self._api_context = ApiContext(
            webhook_port=webhook_port,
            webhook_secret=webhook_secret,
            services_info=services_info or [],
        )

        # Global defaults, preserved so we can restore them when
        # switching away from a configured workspace.
        self._default_model = model
        self._default_timeout = timeout_seconds

        # Apply per-workspace overrides (if configured). Model is
        # validated against codex's CLI surface so a workspaces.yaml
        # entry with `model: gpt-5.4-nano` (valid for goose-on-openai)
        # is silently rejected here. Timeout has no cross-backend
        # equivalent.
        if workspace_config:
            self.model = apply_workspace_model(workspace_config, "codex", self.provider, self.model)
            if workspace_config.timeout is not None:
                self.timeout_seconds = workspace_config.timeout

        # Subprocess and session state
        self._proc: asyncio.subprocess.Process | None = None
        self._session_id: str | None = None
        # time.monotonic() at process start; drives the inherited
        # session-age recycling helpers. Nulled on kill/shutdown.
        self._session_started_at: float | None = None
        self._next_id: int = 1  # Monotonically increasing JSON-RPC request ID
        self._lock = asyncio.Lock()  # Serializes all message sends
        self._fresh_session: bool = True  # True until the first message is sent
        self._stderr_task: asyncio.Task | None = None  # Background stderr drain

        # Cross-user teardown apparatus (#456-equivalent for codex).
        # When effective_codex_user is set in _ensure_started, the
        # subprocess is the sudo wrapper, not codex itself; signalling
        # only the wrapper orphans the codex grandchild because POSIX
        # signal permission rules forbid a service-user-owned killpg
        # from reaching a target-user grandchild. The escalation path
        # uses `sudo -n -u <target> /bin/kill ...` against each cached
        # codex descendant PID. See _send_signal for the full rationale
        # and claude.py for the original implementation this mirrors.
        #
        # The cache is a LIST, ordered innermost-first, because codex's
        # npm-global packaging interposes a node wrapper between sudo
        # and the Rust binary. Killing only the wrapper leaves the Rust
        # binary orphaned to init; signalling the leaf without the
        # wrapper leaves node as a thin shell that normally exits via
        # SIGCHLD but is worth killing explicitly. Both PIDs are owned
        # by the target user; the sudoers rule covers either. A future
        # codex release that drops the node wrapper collapses naturally
        # to a single-PID list with no test-side changes needed beyond
        # the pgrep mocks.
        self._pgid: int | None = None
        self._inner_codex_pids: list[int] = []
        self._effective_codex_user: str | None = None

    @property
    def is_alive(self) -> bool:
        """True if the codex subprocess is running and hasn't exited."""
        return self._proc is not None and self._proc.returncode is None

    @property
    def session_id(self) -> str | None:
        """The current codex session ID, or None if no session is active."""
        return self._session_id

    # ── Process lifecycle ──────────────────────────────────────────

    async def _ensure_started(self) -> None:
        """
        Start the codex app-server subprocess if not already running.

        Launches `codex app-server` and runs the two-step JSON-RPC
        handshake:
        1. initialize - establishes protocol version and capabilities
        2. session/new - creates a session with cwd

        The process persists across prompts. Model selection happens
        via the CODEX_MODEL environment variable (a passthrough of
        self.model; no logical-alias mapping needed at this layer).
        """
        if self.is_alive:
            return

        # Build the subprocess environment. Merge order:
        # 1. Base environment (inherited from parent process)
        # 2. CODEX_MODEL (mirrors self.model)
        # 3. CODEX_PROVIDER (forward-compat placeholder; "openai" today)
        # 4. Per-workspace env_file values
        # 5. Per-workspace inline env values (override env_file)
        # 6. Remove outer control-plane credentials
        # 7. Principal API credential (workspace env can't override it)
        # 8. Per-os-user TMPDIR anchor (cross-user mode only; set
        #    below once the effective user is resolved, LAST so
        #    workspace env cannot point one user's temp writes at
        #    another's)
        env = os.environ.copy()
        if self.model:
            env["CODEX_MODEL"] = self.model
        if self.provider:
            env["CODEX_PROVIDER"] = self.provider
        if self.workspace_config:
            if self.workspace_config.env_file:
                env.update(parse_env_file(self.workspace_config.env_file))
            if self.workspace_config.env:
                env.update(self.workspace_config.env)
        env = sanitize_agent_environment(env)
        if self._api_context.webhook_secret:
            env["KAI_WEBHOOK_SECRET"] = self._api_context.webhook_secret

        # Build the argv. Codex runs either as the bot's process user
        # (single-user install or self-sudo) or as a per-user os_user
        # (multi-user install where each users.yaml entry has its own
        # ~/.codex/auth.json). resolve_claude_user is named claude-
        # historically but its body is backend-agnostic: it returns
        # None when codex_user matches the bot's own user (skipping
        # an unnecessary self-sudo) and the value unchanged otherwise.
        effective_codex_user = resolve_claude_user(self.codex_user)

        # Per-os-user TMPDIR anchor (cross-user mode only). Namespaces
        # every codex temp write under <DATA_DIR>/tmp/<os_user>/ (per-
        # user dir created and chowned by install.py `_apply_migrate`)
        # so two os_users sharing /tmp cannot collide on same-named
        # temp artifacts; the --preserve-env clause below carries
        # TMPDIR through sudo's env_reset. Same posture as the claude
        # and ACP backends. Single-user mode (no codex_user) inherits
        # the system default temp directory.
        if effective_codex_user:
            env["TMPDIR"] = str(DATA_DIR / "tmp" / effective_codex_user)

        # Resolve from the installed backend registry when present.
        # Without a registry, keep the legacy common-path fallback so
        # development and older installs behave unchanged.
        if backend_registry_is_authoritative():
            try:
                codex_bin = resolve_backend_command("codex", allow_bare_fallback=True)
            except BackendRegistryError as e:
                raise RuntimeError(f"Codex startup failed: {e}") from e
        else:
            codex_bin = os.environ.get("CODEX_BIN") or _resolve_default_codex_bin()
        codex_argv = [codex_bin]
        # The reasoning-effort override rides BEFORE the subcommand
        # token: `-c key=value` is defined at the codex CLI root
        # (flattened into the multitool parser and propagated to every
        # subcommand), so root position is the form the CLI's own
        # structure guarantees. Set-or-absent: no flag at all when the
        # knob is empty, leaving the per-OS-user config.toml or the
        # model default in charge of reasoning effort.
        if self.codex_effort_level:
            codex_argv += ["-c", f"model_reasoning_effort={self.codex_effort_level}"]
        codex_argv.append("app-server")
        if effective_codex_user:
            # -H sets HOME to <codex_user>'s pw entry so codex reads
            # auth from ~<codex_user>/.codex/auth.json, not the bot's
            # home. --preserve-env passes the listed vars through
            # sudo's env_reset (the SETENV: sudoers rule allows this).
            # Mirrors claude.py's sudo construction.
            #
            # CODEX_HOME / OPENAI_API_KEY / OPENAI_BASE_URL mirror the
            # one-shot wrap's preserve list in kai.oneshot, so env-key
            # auth and custom-endpoint routing behave the same on the
            # chat leg as on extraction / review / triage for the same
            # users.yaml entry; subscription-OAuth installs carry none
            # of these vars and are unaffected. CODEX_MODEL /
            # CODEX_PROVIDER are deliberately NOT preserved: the
            # authoritative model selection rides the per-turn
            # protocol params, so the env mirrors are informational.
            argv: list[str] = [
                "sudo",
                "-H",
                "-u",
                effective_codex_user,
                "--preserve-env=KAI_WEBHOOK_SECRET,TMPDIR,CODEX_HOME,OPENAI_API_KEY,OPENAI_BASE_URL",
                "--",
            ] + codex_argv
        else:
            argv = codex_argv

        log.info(
            "Starting persistent Codex app-server process (model=%s, provider=%s, user=%s)",
            self.model,
            self.provider,
            effective_codex_user or "(same as bot)",
        )

        self._proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.workspace),
            env=env,
            # Cross-user mode: new session group so the sudo wrapper
            # is the session leader (PGID == PID). os.killpg later
            # reaps the wrapper after the inner-codex sudo escalation
            # in _send_signal / _async_send_signal_for_close has
            # signalled the grandchild directly. Without the
            # escalation, killpg alone would orphan the codex
            # grandchild (POSIX permission rules: service user
            # cannot signal a process whose UIDs are all the
            # target user). See _send_signal for the full chain.
            start_new_session=bool(effective_codex_user),
            # A single codex event can carry the full text of a tool
            # call's output inline (e.g. a `gh pr diff` result on a
            # reasoning item, or an item/completed for a long
            # agentMessage). The asyncio.StreamReader default limit
            # is 64KB; we previously bumped it to 1MB which was still
            # too tight for PR-review-sized chunks - real codex review
            # turns surfaced "Separator is not found, and chunk exceed
            # the limit" from readline. 16MB is well above any
            # plausible single-event payload while still bounding
            # memory if codex ever produces a runaway line. A
            # proper streaming reader (chunked + reassemble on \n)
            # would remove the ceiling entirely; deferred until the
            # 16MB ceiling is itself observed in practice.
            limit=16 * 1024 * 1024,
        )
        self._session_started_at = time.monotonic()
        self._stderr_task = asyncio.create_task(self._drain_stderr())

        # Save the process group ID for reliable signal delivery.
        # When effective_codex_user is set, start_new_session=True
        # made the wrapper a session leader so PGID == PID. We save
        # it now because os.getpgid() fails after the process exits,
        # but os.killpg() works as long as any group member survives.
        if effective_codex_user:
            self._pgid = self._proc.pid
        else:
            self._pgid = None

        # Remember the resolved sudo target for the #456-equivalent
        # cross-user signal escalation. _inner_codex_pids is reset to
        # an empty list here so a recycled instance does not carry
        # stale PIDs from a previous spawn into the fresh tree.
        self._effective_codex_user = effective_codex_user
        self._inner_codex_pids = []

        # Handshake per the codex app-server protocol:
        #   1. Client `initialize` request (clientInfo + optional
        #      capabilities). NO protocolVersion field; the server
        #      reports the protocol it speaks via its response and
        #      via per-method error messages, not a version echo.
        #   2. Server response: userAgent, codexHome, platformFamily,
        #      platformOs. We do not need any of these at runtime;
        #      reading the result is purely a handshake gate.
        #   3. Client `initialized` notification (no id). Required
        #      before any other request on the connection. Skipping
        #      this would have all subsequent calls rejected with
        #      "Not initialized".
        #   4. Client `thread/start` request. Returns a thread object
        #      whose `id` field is the persistent session handle the
        #      bot uses for every subsequent turn/start.
        # The optOutNotificationMethods list suppresses streams the
        # bot does not consume - remoteControl status pings, MCP
        # startup chatter, and thread/started which we already get
        # via the thread/start response itself.
        self._next_id = 1
        await self._write_rpc(
            "initialize",
            {
                "clientInfo": {"name": "kai", "version": kai.__version__},
                "capabilities": {
                    "optOutNotificationMethods": [
                        "remoteControl/status/changed",
                        "mcpServer/startupStatus/updated",
                        "thread/started",
                        "thread/tokenUsage/updated",
                    ],
                },
            },
        )
        await self._read_result(expected_id=1)

        await self._write_notification("initialized")

        # `approvalPolicy: "never"` is load-bearing for an unattended
        # Telegram bot. Codex's default is "on-request", which makes
        # the server emit approval-request notifications and wait for
        # a client response before any tool call. Kai has no human
        # in the loop to approve from Telegram, so on-request gates
        # silently: codex waits forever, the bot's stdout-readline
        # ceiling fires, the operator sees "Codex timed out".
        #
        # `sandbox: "danger-full-access"` matches the claude backend's
        # posture: claude --print runs unsandboxed with whatever
        # permissions the bot's os_user has. The `workspace-write`
        # alternative would disable network access (codex's default
        # for workspace-write profiles), breaking `gh`, `curl`, `pip`,
        # and anything else the bot needs to reach outside the local
        # filesystem. The bot already runs under a per-user sudo wrap
        # with that user's full file authority; constraining codex
        # tighter than the surrounding process makes no security
        # sense.
        #
        # Sandbox variants are kebab-case (`read-only`, `workspace-write`,
        # `danger-full-access`); the codex server rejects camelCase
        # spellings at thread/start with "unknown variant".
        thread_params: dict = {
            "cwd": str(self.workspace),
            "approvalPolicy": "never",
            "sandbox": "danger-full-access",
        }
        if self.model:
            thread_params["model"] = self.model
        await self._write_rpc("thread/start", thread_params)
        result = await self._read_result(expected_id=2)

        # The thread.id is the conversational handle for all
        # subsequent turn/start calls. We reuse the existing
        # self._session_id attribute (ABC-mandated) to hold it;
        # naming stays "session_id" for cross-backend consistency
        # but the value is codex's thread UUID.
        thread = result.get("thread", {})
        thread_id = thread.get("id")
        if not thread_id:
            raise RuntimeError(
                "Codex thread/start returned no thread.id; the pinned codex CLI may have an incompatible schema."
            )
        self._session_id = thread_id
        self._fresh_session = True

    async def _drain_stderr(self) -> None:
        """
        Continuously read and discard stderr from the codex process.

        Without this, the stderr pipe buffer fills up and the process
        deadlocks. Lines are logged at DEBUG level for diagnostics.
        """
        while self._proc and self._proc.stderr:
            try:
                line = await self._proc.stderr.readline()
                if not line:
                    break
                text = line.decode().strip()
                if text:
                    log.debug("Codex stderr: %s", text[:200])
            except Exception:
                log.warning("Unexpected error in Codex stderr drain", exc_info=True)
                break

    # ── JSON-RPC helpers ───────────────────────────────────────────

    async def _write_rpc(self, method: str, params: dict) -> int:
        """
        Write a JSON-RPC 2.0 request to the subprocess stdin.

        Increments the monotonic request ID, serializes the message
        with a trailing newline, and flushes. Returns the request id
        so callers can correlate the response. Raises if the process
        or its stdin pipe is gone.
        """
        assert self._proc is not None and self._proc.stdin is not None
        request_id = self._next_id
        self._next_id += 1
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
        self._proc.stdin.write(msg.encode())
        await self._proc.stdin.drain()
        return request_id

    async def _write_notification(self, method: str, params: dict | None = None) -> None:
        """
        Write a JSON-RPC 2.0 notification (no id, no response expected).

        Used for the `initialized` handshake step: the codex app-server
        spec requires the client to send `initialized` between the
        `initialize` response and any other request on that connection.
        """
        assert self._proc is not None and self._proc.stdin is not None
        body: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            body["params"] = params
        self._proc.stdin.write((json.dumps(body) + "\n").encode())
        await self._proc.stdin.drain()

    async def _read_result(self, expected_id: int) -> dict:
        """
        Read stdout lines until a JSON-RPC result with the expected id appears.

        Discards session/update notifications that codex may emit
        during startup. Raises on JSON-RPC error responses or timeout.
        """
        assert self._proc is not None and self._proc.stdout is not None
        while True:
            try:
                line = await asyncio.wait_for(
                    self._proc.stdout.readline(),
                    timeout=self.timeout_seconds,
                )
            except TimeoutError as exc:
                raise TimeoutError(f"Codex handshake timed out waiting for response id={expected_id}") from exc

            if not line:
                raise RuntimeError("Codex process exited during handshake")

            try:
                msg = json.loads(line.decode())
            except json.JSONDecodeError:
                # Non-JSON output during startup (e.g., progress bars)
                continue

            # Check for matching response
            if msg.get("id") == expected_id:
                if "error" in msg:
                    err = msg["error"]
                    raise RuntimeError(f"Codex error (id={expected_id}): {err.get('message', 'unknown error')}")
                return msg.get("result", {})

            # Discard notifications during handshake

    # ── Sending prompts ────────────────────────────────────────────

    async def send(self, prompt: str | list, chat_id: int | None = None) -> AsyncIterator[StreamEvent]:
        """
        Send a message to codex and yield streaming events.

        Serialized via an internal lock so concurrent callers queue
        rather than interleave. Context injection (identity, memory,
        history, API docs) is prepended on the first message of each
        session, identical to ClaudeCodeBackend and GooseBackend.

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

        Handles context injection, prompt coercion to the JSON-RPC
        content-block format, session/prompt dispatch, and streaming
        response accumulation from session/update notifications.
        """
        # Recycle the session if it has exceeded the configured age
        # limit. Prevents unbounded memory growth in a long-lived
        # agent subprocess. Checked only before starting a new
        # interaction, never during one, so in-flight responses
        # complete normally. No save-prompt equivalent exists on the
        # codex CLI; the kill is immediate, and _ensure_started()
        # below respawns with a fresh handshake (which also re-marks
        # the session fresh for context injection).
        if self._should_recycle():
            log.info(
                "Codex session age %.1f hours exceeds limit of %.1f hours; recycling",
                self._session_age_hours(),
                self.max_session_hours,
            )
            await self._kill()

        try:
            await self._ensure_started()
        except (OSError, RuntimeError, TimeoutError) as exc:
            yield StreamEvent(
                text_so_far="",
                done=True,
                response=AgentResponse(success=False, text="", error=f"Codex startup failed: {exc}"),
            )
            return

        # Per-turn prompt context. Build the fresh-session bootstrap
        # (MEMORY.md / PREFERENCES.md seed + session_context block)
        # and the foreign-workspace reminder LOCALLY, then hand both
        # to `assemble_turn_context` so the shared helper owns the
        # ordering invariant (USER_MESSAGE_MARKER closest to user
        # text; session_context, semantic memory, workspace reminder
        # stacked above in that order). Mirrors the ClaudeCodeBackend
        # wire-through; backend-owned lifecycle stays here, shared
        # string composition stays in the helper.
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

        # Foreign-workspace reminder is built fresh per turn (its
        # value depends on the current workspace, which the operator
        # can change between messages via `/workspace`).
        # `assemble_turn_context` no-ops on an empty string, so
        # coerce the None to "" rather than threading the optional
        # through the helper signature.
        reminder = build_foreign_workspace_reminder(self.workspace, self.home_workspace) or ""

        # Normalize user blocks to the app-server `UserInput` shape
        # before per-turn assembly. Text blocks pass through; image
        # blocks are materialized to disk and forwarded as
        # `localImage` items (the protocol schema's image carrier; no
        # capability gate exists or is needed, the schema is fixed by
        # the installed CLI). A block that cannot be materialized
        # (wrong shape, bad base64, write failure) is dropped and
        # counted so the reply carries a user-visible notice; a
        # log-only drop leaves the user believing the model saw an
        # image it never received. Normalization must run BEFORE
        # assemble_turn_context so the marker it prepends labels a
        # real user region. Running it after the helper would leave
        # injected text layers (session_context, reminder, memory)
        # above the marker and nothing below it.
        had_user_text = isinstance(prompt, str)
        dropped_images = 0
        turn_image_paths: list[Path] = []
        if isinstance(prompt, list):
            input_blocks: list[dict] = []
            for block in prompt:
                block_type = block.get("type")
                if block_type == "text":
                    input_blocks.append({"type": "text", "text": block["text"]})
                    continue
                if block_type == "image":
                    image_path = write_turn_image_file(
                        block,
                        chat_id=chat_id,
                        reader_user=self._effective_codex_user,
                    )
                    if image_path is not None:
                        turn_image_paths.append(image_path)
                        input_blocks.append({"type": "localImage", "path": str(image_path)})
                        continue
                    dropped_images += 1
                log.warning(
                    "CodexBackend: dropping content block type=%s",
                    block_type,
                )
            had_user_text = any(b.get("type") == "text" for b in input_blocks)
            # Codex requires a non-empty `input` array. Input whose
            # every block was dropped becomes a single placeholder so
            # the marker has a user region to label.
            prompt = input_blocks or [{"type": "text", "text": "(empty prompt)"}]

        # `chat_id=None` to the helper suppresses semantic recall for
        # this turn. The placeholder text "(empty prompt)" is backend-
        # synthetic and must not become a memory search query; only
        # real user text drives recall. Session context still uses
        # the real chat_id - it was built above this point.
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

        # Coerce to the JSON-RPC content-block shape. `prompt` is
        # either a str (from a str input; the helper preserves the
        # input type family) or a list already normalized to
        # `UserInput` items above (text plus any localImage entries).
        rpc_prompt: list[dict]
        if isinstance(prompt, str):
            rpc_prompt = [{"type": "text", "text": prompt}]
        else:
            rpc_prompt = prompt

        # Send the turn/start request. The codex app-server protocol
        # uses `input` (array of typed content blocks) plus `threadId`;
        # the JSON-RPC response carries the new turn's id, and the
        # actual model output streams as item/* and turn/* notifications
        # until a final turn/completed arrives.
        assert self._proc is not None
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None

        try:
            turn_params: dict = {
                "threadId": self._session_id,
                "input": rpc_prompt,
            }
            # Pin model per-turn so workspace_config overrides or
            # /model switches take effect on the next message without
            # restarting the thread.
            if self.model:
                turn_params["model"] = self.model
            prompt_id = await self._write_rpc("turn/start", turn_params)
        except OSError as e:
            log.error("Failed to write to Codex process: %s", e)
            await self._kill()
            # The turn never dispatched, so this turn's image files
            # have no reader; the streaming loop's cleanup is not
            # reached on this path.
            _unlink_turn_images(turn_image_paths)
            yield StreamEvent(
                text_so_far="",
                done=True,
                response=AgentResponse(
                    success=False,
                    text="",
                    error="Codex process died, restarting on next message",
                ),
            )
            return

        # Stream response: read stdout lines, accumulate text chunks,
        # and yield StreamEvents. Unknown event types are logged at
        # DEBUG and skipped (schema-drift defense).
        #
        # Multi-agentMessage tracking: a single turn can emit MORE
        # than one agentMessage item (e.g. preamble, tool call,
        # post-tool summary). Per the codex protocol README, deltas
        # are scoped to a single itemId; concatenating them across
        # items without separators tacks the second item's first
        # word onto the first item's terminator ("summary here.The"
        # -> the operator's smoke-test observation). We track each
        # item's text separately, commit completed items into a
        # joined-with-blank-line prefix, and only let item/completed
        # override the CURRENT item's text - never the prior
        # committed content. The visible text streamed to telegram
        # is `committed + ("\n\n" + current if current)`.
        # When images were dropped above, the committed prefix is
        # seeded with a notice so the user learns the model never saw
        # them; the drop is otherwise invisible (the reply reads as a
        # normal answer to the caption text). The seed carries no
        # trailing separator because every commit join below inserts
        # "\n\n" after a non-empty committed prefix. `got_model_text`
        # exists because the seed makes the visible text non-empty
        # before the model says anything; the EOF branch below must
        # judge "did the model produce output" from this flag, not
        # from visible-text truthiness, or a process that dies
        # silently after an image drop would surface as a successful
        # notice-only reply.
        committed_text = ""
        got_model_text = False
        if dropped_images:
            noun = "image" if dropped_images == 1 else "images"
            committed_text = (
                f"[Note: {dropped_images} attached {noun} could not be passed to "
                f"Codex; this reply is based on the message text only.]"
            )
        current_item_id: str | None = None
        current_item_text = ""

        def _visible_text() -> str:
            if not current_item_id:
                return committed_text
            if not committed_text:
                return current_item_text
            return committed_text + "\n\n" + current_item_text

        last_activity = time.monotonic()
        max_idle_seconds = self.timeout_seconds * 5

        try:
            while True:
                # Check idle timeout before each readline
                idle = time.monotonic() - last_activity
                if idle > max_idle_seconds:
                    log.error(
                        "Codex idle timeout (%.0fs with no output, limit %ds)",
                        idle,
                        max_idle_seconds,
                    )
                    await self._kill()
                    visible = _visible_text()
                    yield StreamEvent(
                        text_so_far=visible,
                        done=True,
                        response=AgentResponse(
                            success=False,
                            text=visible,
                            error="Codex interaction timed out (no output)",
                        ),
                    )
                    return

                try:
                    line = await asyncio.wait_for(
                        self._proc.stdout.readline(),
                        timeout=self.timeout_seconds * 3,
                    )
                except TimeoutError:
                    log.error("Codex response timed out")
                    await self._kill()
                    visible = _visible_text()
                    yield StreamEvent(
                        text_so_far=visible,
                        done=True,
                        response=AgentResponse(
                            success=False,
                            text=visible,
                            error="Codex timed out",
                        ),
                    )
                    return

                # Reset idle timer on non-empty output
                if line:
                    last_activity = time.monotonic()
                else:
                    # EOF - process died. Success iff the model said
                    # anything; the notice seed alone does not count
                    # (see `got_model_text` above).
                    log.error("Codex process EOF")
                    await self._kill()
                    visible = _visible_text()
                    yield StreamEvent(
                        text_so_far=visible,
                        done=True,
                        response=AgentResponse(
                            success=got_model_text,
                            text=visible,
                            error=None if got_model_text else "Codex process ended unexpectedly",
                        ),
                    )
                    return

                try:
                    msg = json.loads(line.decode())
                except json.JSONDecodeError:
                    log.debug("Skipping non-JSON line: %s", line[:200])
                    continue

                # Codex app-server event vocabulary (per the protocol
                # README at codex-rs/app-server/README.md):
                #
                #   - turn/start RESPONSE: matched by id, returns the
                #     turn object. We acknowledge it but do not finish
                #     here; the turn is still streaming.
                #   - turn/started NOTIFICATION: opted out via
                #     initialize.capabilities so it never arrives.
                #   - item/started: full ThreadItem with type and id.
                #     We only care about agentMessage items for the
                #     conversational stream; other item types (reasoning,
                #     commandExecution, fileChange, etc.) are logged at
                #     DEBUG. We capture the agentMessage item id so
                #     subsequent deltas can be tied back to the right
                #     stream.
                #   - item/agentMessage/delta: streaming text chunks.
                #     Concatenate `delta` per itemId. Codex emits text
                #     in roughly token-sized chunks.
                #   - item/completed: authoritative final state of the
                #     item. For agentMessage this carries the full
                #     accumulated `text`; we trust this over our own
                #     concatenation in case any delta was missed.
                #   - turn/completed: terminal event. Carries turn
                #     status (`completed` / `interrupted` / `failed`)
                #     and an optional error payload on failure. This
                #     is the signal to yield the final StreamEvent.
                #   - error notification: mid-turn error; may precede
                #     a failed turn/completed. We treat it as terminal.
                method = msg.get("method")
                if method == "item/started":
                    item = msg.get("params", {}).get("item", {})
                    if item.get("type") == "agentMessage":
                        # Begin a new in-flight agentMessage. Anything
                        # currently in current_item_text was uncommitted
                        # (no item/completed arrived); commit it now
                        # so the new item's deltas start fresh and we
                        # don't lose mid-stream text on the boundary.
                        if current_item_id and current_item_text:
                            committed_text = (
                                committed_text + "\n\n" + current_item_text if committed_text else current_item_text
                            )
                        current_item_id = item.get("id")
                        current_item_text = ""
                    else:
                        log.debug("Codex: item/started type=%s id=%s", item.get("type"), item.get("id"))

                elif method == "item/agentMessage/delta":
                    params = msg.get("params", {})
                    delta_text = params.get("delta", "")
                    delta_item_id = params.get("itemId")
                    if delta_text:
                        got_model_text = True
                        # Defensive: if a delta arrives without a
                        # prior item/started (out-of-order or schema
                        # drift), treat it as opening a new item.
                        if delta_item_id and delta_item_id != current_item_id:
                            if current_item_id and current_item_text:
                                committed_text = (
                                    committed_text + "\n\n" + current_item_text if committed_text else current_item_text
                                )
                            current_item_id = delta_item_id
                            current_item_text = ""
                        current_item_text += delta_text
                        yield StreamEvent(text_so_far=_visible_text())

                elif method == "item/completed":
                    item = msg.get("params", {}).get("item", {})
                    if item.get("type") == "agentMessage":
                        # Authoritative final text for THIS item only.
                        # Override current_item_text if codex's final
                        # value differs from our delta sum (the docs
                        # call item/completed the source of truth).
                        final_text = item.get("text", "")
                        if final_text:
                            current_item_text = final_text
                        # Commit the in-flight item to the prefix and
                        # reset. Subsequent items append after a blank
                        # line; subsequent deltas can never overwrite
                        # this text. `got_model_text` is set here as
                        # well as on deltas because an item/completed
                        # can carry text that never streamed as deltas.
                        if current_item_text:
                            got_model_text = True
                            committed_text = (
                                committed_text + "\n\n" + current_item_text if committed_text else current_item_text
                            )
                        current_item_id = None
                        current_item_text = ""
                        yield StreamEvent(text_so_far=_visible_text())
                    else:
                        log.debug("Codex: item/completed type=%s", item.get("type"))

                elif method == "turn/completed":
                    turn = msg.get("params", {}).get("turn", {})
                    status = turn.get("status")
                    # Flush any uncommitted current_item_text so the
                    # final response carries it (schema-drift defense:
                    # a missing item/completed should not lose text).
                    if current_item_id and current_item_text:
                        committed_text = (
                            committed_text + "\n\n" + current_item_text if committed_text else current_item_text
                        )
                        current_item_id = None
                        current_item_text = ""
                    final_visible = committed_text
                    if status == "completed":
                        yield StreamEvent(
                            text_so_far=final_visible,
                            done=True,
                            response=AgentResponse(
                                success=True,
                                text=final_visible,
                                session_id=self._session_id,
                                duration_ms=0,
                            ),
                        )
                        return
                    # Non-completed terminal: interrupted or failed.
                    # turn.error carries the diagnostic when present.
                    err_obj = turn.get("error") or {}
                    err_msg = err_obj.get("message") or f"Codex turn ended with status={status}"
                    yield StreamEvent(
                        text_so_far=final_visible,
                        done=True,
                        response=AgentResponse(
                            success=False,
                            text=final_visible,
                            error=err_msg,
                        ),
                    )
                    return

                elif method == "error":
                    # Mid-turn error notification. Treat as terminal;
                    # the subsequent turn/completed with status=failed
                    # would be redundant.
                    err_obj = msg.get("params", {}).get("error", {})
                    err_msg = err_obj.get("message") or "Codex error"
                    visible = _visible_text()
                    yield StreamEvent(
                        text_so_far=visible,
                        done=True,
                        response=AgentResponse(
                            success=False,
                            text=visible,
                            error=err_msg,
                        ),
                    )
                    return

                # Response to our turn/start (id-matched). The result
                # carries the initial turn object; we acknowledge it
                # but the stream continues until turn/completed.
                elif msg.get("id") == prompt_id and "result" in msg:
                    log.debug("Codex: turn/start acknowledged for id=%s", prompt_id)

                # JSON-RPC error matched on our turn/start id - request
                # never made it to streaming, so finish here.
                elif msg.get("id") == prompt_id and "error" in msg:
                    err = msg["error"].get("message", "unknown codex error")
                    visible = _visible_text()
                    yield StreamEvent(
                        text_so_far=visible,
                        done=True,
                        response=AgentResponse(
                            success=False,
                            text=visible,
                            error=err,
                        ),
                    )
                    return

                else:
                    # Unknown method or unmatched id. Schema-drift
                    # defense: a new codex release adding extra
                    # notification types must not break the
                    # conversational stream.
                    log.debug(
                        "Codex: skipping unrecognized message id=%s method=%s",
                        msg.get("id"),
                        method,
                    )

        except Exception as e:
            log.exception("Unexpected error reading Codex stream")
            await self._kill()
            visible = _visible_text()
            yield StreamEvent(
                text_so_far=visible,
                done=True,
                response=AgentResponse(
                    success=False,
                    text=visible,
                    error=str(e),
                ),
            )
        finally:
            # Every exit from the loop above is terminal for the turn
            # (completion, turn failure, timeout, EOF, mid-turn error,
            # or the consumer closing the generator), and codex
            # consumes image input when the turn is processed, so the
            # files have no reader past this point.
            _unlink_turn_images(turn_image_paths)

    # ── Kill / restart / shutdown ──────────────────────────────────

    def _lookup_inner_codex_pids(self) -> list[int]:
        """
        Find the codex descendant PIDs in cross-user mode.

        In cross-user mode the bot spawns `sudo -u <target> -- codex
        app-server` and self._proc tracks the sudo wrapper. Codex's
        npm-global packaging interposes a node wrapper between sudo
        and the actual Rust binary, so the process tree is:

            sudo (service user)
              └── node /path/to/codex app-server  (target user)
                    └── /path/.../codex/codex app-server  (target user)

        `pgrep -P <sudo_pid>` returns the node PID. We then `pgrep -P`
        against that to find the Rust binary. Returns the descendant
        PIDs in inner-to-outer order: `[rust_pid, node_pid]`. When
        a future codex release drops the node wrapper (single-binary
        install), the second pgrep returns nothing and the result
        collapses to `[node_pid]` (which IS the Rust binary in that
        case). Returns an empty list when no sudo wrapper is alive,
        sudo has not yet forked, or pgrep fails outright.

        Innermost-first ordering matters: _send_signal iterates the
        list in order so the Rust binary dies before the node wrapper
        gets killpg'd (the wrapper would otherwise exit via SIGCHLD
        regardless, but signalling the Rust binary first is the
        defensive shape that catches the original #487 orphan).

        Synchronous variant used by the sync force_kill path.
        Mirrors claude.py's _lookup_inner_claude_pid (#456/#459), with
        the extra level for codex's npm packaging that #487 surfaced.
        """
        if self._proc is None:
            return []
        first_pid = self._pgrep_first_child(self._proc.pid)
        if first_pid is None:
            return []
        # Second level: pgrep the first child to find its child (the
        # Rust binary under the node wrapper). If pgrep returns
        # nothing the install has no wrapper layer; the first PID is
        # itself the leaf.
        second_pid = self._pgrep_first_child(first_pid)
        if second_pid is None:
            return [first_pid]
        return [second_pid, first_pid]

    @staticmethod
    def _pgrep_first_child(parent_pid: int) -> int | None:
        """Run pgrep -P <parent_pid> and return the first child PID,
        or None if pgrep fails / returns no children. Extracted so
        the two-level walk in _lookup_inner_codex_pids reads as one
        idea per line."""
        try:
            result = subprocess.run(
                ["pgrep", "-P", str(parent_pid)],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError):
            return None
        if result.returncode != 0:
            return None
        first_line = result.stdout.strip().split("\n", 1)[0] if result.stdout.strip() else ""
        if not first_line:
            return None
        try:
            return int(first_line)
        except ValueError:
            return None

    async def _async_lookup_inner_codex_pids(self) -> list[int]:
        """
        Async pgrep equivalent of `_lookup_inner_codex_pids`.

        Same semantics and two-level walk, but uses
        asyncio.create_subprocess_exec so the event loop is not
        blocked while pgrep runs. Called from the async _kill /
        shutdown paths; the sync variant remains for force_kill.
        2s ceiling matches the sync version. Mirrors claude.py's
        _async_lookup_inner_claude_pid (#459), extended to walk
        the extra level for codex's npm packaging (#487).
        """
        if self._proc is None:
            return []
        first_pid = await self._async_pgrep_first_child(self._proc.pid)
        if first_pid is None:
            return []
        second_pid = await self._async_pgrep_first_child(first_pid)
        if second_pid is None:
            return [first_pid]
        return [second_pid, first_pid]

    @staticmethod
    async def _async_pgrep_first_child(parent_pid: int) -> int | None:
        """Async pgrep -P <parent_pid>, returning the first child PID
        or None. Companion to _pgrep_first_child; same contract."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "pgrep",
                "-P",
                str(parent_pid),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except (OSError, FileNotFoundError):
            return None
        try:
            stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=2)
        except TimeoutError:
            # pgrep hung; reap it so we do not leak a child. The
            # kill() guard handles the race where pgrep exited
            # between wait_for timing out and the kill() call:
            # ProcessLookupError is a benign already-dead signal.
            try:
                proc.kill()
            except OSError:
                pass
            try:
                await proc.wait()
            except OSError:
                pass
            return None
        if proc.returncode != 0:
            return None
        first_line = stdout_bytes.decode(errors="replace").strip().split("\n", 1)[0]
        if not first_line:
            return None
        try:
            return int(first_line)
        except ValueError:
            return None

    async def _async_sudo_kill(self, target_user: str, pid: int, sig: int) -> None:
        """
        Signal one inner codex PID through the canonical cross-user
        helper in kai.acp.

        Thin delegation seam kept as a method so the kill-ordering
        tests can patch one name on the instance. The helper owns
        the `sudo -n -u <target> /bin/kill -<sig> <pid>` spawn, the
        bounded wait, and the rc/ESRCH classification: the benign
        already-exited race (inner codex died between the PID cache
        time and signal delivery) logs at DEBUG, while real failures
        (sudoers rule missing, signal permission denied, kill binary
        missing) reach the WARNING stream operators watch.
        """
        await _signal_target_user_pid(
            target_user=target_user,
            pid=pid,
            sig=sig,
            purpose="chat",
            backend=self.backend_name,
        )

    def _send_signal(self, sig: int) -> None:
        """
        Send a signal to the codex process (or process group + sudo
        escalation when running cross-user).

        Cross-user mode: os.killpg from the service user cannot signal
        a target-user codex grandchild because POSIX signal permission
        rules require real or effective UID to match real or saved
        set-user-ID of the target. killpg succeeds for the sudo
        wrapper (real UID is the service user) but the inner codex
        (UIDs = target) survives and gets orphaned to init. We
        escalate via `sudo -n -u <target> /bin/kill -<sig> <pid>` for
        the inner codex BEFORE killpg reaps the sudo wrapper.
        install.py's _generate_sudoers emits a per-target
        NOPASSWD: /bin/kill rule for this path.

        Mirrors claude.py's _send_signal (#456/#458). Deliberately
        does NOT check self._proc.returncode: when codex_user is set
        self._proc tracks the sudo wrapper, not codex, and checking
        returncode would skip delivery if sudo exited first.
        """
        if self._pgid is not None:
            if self._effective_codex_user is not None:
                if not self._inner_codex_pids:
                    self._inner_codex_pids = self._lookup_inner_codex_pids()
                # Innermost-first: kill the Rust binary before the
                # node wrapper so the wrapper does not get a chance
                # to spawn a fresh leaf, and so the failure-to-orphan
                # diagnostic in the warning identifies the load-bearing
                # PID. A future codex release that drops the wrapper
                # collapses to a single-element list and the loop
                # body runs exactly once. _inner_codex_pids is set up
                # in inner-to-outer order by _lookup_inner_codex_pids.
                for pid in self._inner_codex_pids:
                    # Canonical sync helper: owns the absolute
                    # /bin/kill argv (matching the sudoers rule
                    # install.py pins), the bounded wait, and the
                    # rc/ESRCH classification. Failures are logged
                    # and swallowed inside the helper, so the loop
                    # always proceeds to the next PID and then to
                    # killpg - the wrapper still needs reaping
                    # regardless of how each inner-PID signal went.
                    _signal_target_user_pid_sync(
                        target_user=self._effective_codex_user,
                        pid=pid,
                        sig=int(sig),
                        purpose="chat",
                        backend=self.backend_name,
                    )
            try:
                os.killpg(self._pgid, sig)
            except OSError:
                pass
        elif self._proc is not None:
            try:
                self._proc.send_signal(sig)
            except OSError:
                pass

    async def _async_send_signal_for_close(self, sig: int) -> None:
        """
        Async equivalent of `_send_signal` used by `_kill` and `shutdown`.

        Same three-step structure as the sync version: prime the
        inner-codex-PID cache (async pgrep), signal the inner codex
        through sudo (async subprocess), then reap the wrapper via
        os.killpg (a non-blocking syscall, sync is fine). The
        single-user fallback uses _proc.send_signal (also non-
        blocking). Mirrors claude.py's _async_send_signal_for_close.
        """
        if self._pgid is not None:
            if self._effective_codex_user is not None:
                if not self._inner_codex_pids:
                    self._inner_codex_pids = await self._async_lookup_inner_codex_pids()
                # Innermost-first; see _send_signal for the full
                # rationale on ordering and the npm-wrapper failure
                # mode this guards against.
                for pid in self._inner_codex_pids:
                    await self._async_sudo_kill(
                        self._effective_codex_user,
                        pid,
                        int(sig),
                    )
            try:
                os.killpg(self._pgid, sig)
            except OSError:
                pass
        elif self._proc is not None:
            try:
                self._proc.send_signal(sig)
            except OSError:
                pass

    def force_kill(self) -> None:
        """
        Kill the subprocess immediately via SIGKILL.

        Safe to call without holding the lock. Called by /stop to abort
        an in-flight response. Cross-user mode goes through
        _send_signal so the codex grandchild is sudo-escalated before
        the wrapper is killpg'd, avoiding the #456 orphan leak.
        Single-user mode falls back to _proc.kill() via the
        _send_signal else-branch. Does NOT null _proc - full cleanup
        happens in _kill() when the read loop detects EOF.
        """
        if self._proc is not None:
            self._send_signal(signal.SIGKILL)
        if self._stderr_task:
            self._stderr_task.cancel()
            self._stderr_task = None

    async def _kill(self) -> None:
        """
        Kill the subprocess and clean up all process state.

        Sends SIGKILL via the async escalation path (so cross-user
        installs do not orphan the codex grandchild), waits up to 5
        seconds for exit, then nulls all process references.
        Idempotent.
        """
        if self._proc:
            await self._async_send_signal_for_close(signal.SIGKILL)
            if self._stderr_task:
                self._stderr_task.cancel()
                self._stderr_task = None
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except TimeoutError:
                pass
            self._proc = None
            self._session_id = None
            self._pgid = None
            self._inner_codex_pids = []
            self._effective_codex_user = None
            self._session_started_at = None

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

        Kills the current subprocess. The next send() restarts codex
        in the new directory with the new config applied.
        """
        await self._kill()
        self.workspace = new_workspace
        self.workspace_config = workspace_config

        # Revert to global defaults, then apply overrides. Prevents
        # stale values when switching from a fully-configured workspace
        # to a partially-configured one. Model override validated
        # against codex's CLI surface.
        self.model = self._default_model
        self.timeout_seconds = self._default_timeout
        if workspace_config:
            self.model = apply_workspace_model(workspace_config, "codex", self.provider, self.model)
            if workspace_config.timeout is not None:
                self.timeout_seconds = workspace_config.timeout
        # _fresh_session is set True by _ensure_started() on next send()

    async def shutdown(self) -> None:
        """
        Graceful shutdown. No save prompt - codex CLI has no equivalent.

        Sends SIGTERM first via the async escalation path so the
        target-user codex grandchild gets sudo-signalled before the
        wrapper is reaped; waits up to 5 seconds for clean exit.
        Falls back to SIGKILL (also through the escalation path) if
        the process doesn't terminate in time.
        """
        if self._proc:
            await self._async_send_signal_for_close(signal.SIGTERM)
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except TimeoutError:
                await self._async_send_signal_for_close(signal.SIGKILL)
                try:
                    await asyncio.wait_for(self._proc.wait(), timeout=5)
                except TimeoutError:
                    log.warning("CodexBackend: process did not exit after SIGKILL")
        if self._stderr_task:
            self._stderr_task.cancel()
            self._stderr_task = None
        self._proc = None
        self._session_id = None
        self._pgid = None
        self._inner_codex_pids = []
        self._effective_codex_user = None
        self._session_started_at = None
