"""
Claude Code subprocess backend.

Implements the AgentBackend ABC for Claude Code's stream-json protocol.
Manages a long-running subprocess that accepts prompts on stdin and
streams responses on stdout as newline-delimited JSON.

This is the concrete backend that pool.py instantiates by default.
Process management (spawn, stream, kill, restart) lives here; context
injection (identity, memory, history, API docs) lives in backend.py
as shared functions usable by any backend.

The stream-json protocol:
    Input:  {"type": "user", "message": {"role": "user", "content": [...]}}
    Output: {"type": "system", ...}      - session metadata
            {"type": "assistant", ...}   - partial text (streaming)
            {"type": "result", ...}      - final response with cost/session info
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
import time
from collections.abc import AsyncIterator
from pathlib import Path

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


# ── Claude Code backend ─────────────────────────────────────────────


class ClaudeCodeBackend(AgentBackend):
    """
    AgentBackend implementation for Claude Code's stream-json protocol.

    Manages the lifecycle of a Claude Code subprocess: starting, sending
    messages, streaming responses, killing/restarting, and workspace
    switching. All message sends are serialized via an internal asyncio
    lock to prevent interleaving.

    The process runs with --permission-mode bypassPermissions (required
    for headless operation via Telegram). No --max-budget-usd is emitted:
    the claude backend is committed to Max-plan OAuth, where the CLI's
    computed-cost ceiling has no relation to actual billing (the CLI
    tracks pay-per-token rates whether or not the operator owes any of
    that money). Terminating the subprocess at a phantom ceiling would
    just stop work that is not costing anything. Runaway protection
    comes from the per-message stdout-read timeouts and idle detection
    (both derived from `timeout_seconds`) - a stuck or recursive
    subprocess surfaces via stream-read failure rather than holding
    the executor indefinitely.
    """

    def __init__(
        self,
        *,
        model: str = "sonnet",
        workspace: Path = Path("home"),
        home_workspace: Path | None = None,
        webhook_port: int = 8080,
        webhook_secret: str = "",
        max_budget_usd: float = 1.0,
        timeout_seconds: int = 120,
        services_info: list[dict] | None = None,
        claude_user: str | None = None,
        max_session_hours: float = 0,
        workspace_config: WorkspaceConfig | None = None,
        max_context_window: int = 0,
        autocompact_pct: int = 0,
        # Default duplicated from Config.claude_effort_level (config.py).
        # Production calls (pool.py) always pass an explicit value from
        # config so the duplication is invisible there; the default here
        # exists so direct ClaudeCodeBackend(...) instantiation in tests
        # does not have to plumb an effort kwarg. Keep the two in sync.
        claude_effort_level: str = "high",
        # Operator-intent flag for the memory subsystem (Config.
        # memory_enabled). Drives the [Memory subsystem: ...] marker
        # emission and gates MEMORY.md inject in build_session_context.
        # Default False so direct test instantiations need not plumb
        # the kwarg; production callers (pool.py) always pass an
        # explicit value.
        memory_enabled: bool = False,
    ):
        # ABC-required attributes (pool.py reads/writes these)
        self.model = model
        self.workspace = workspace
        self.home_workspace = home_workspace or workspace
        self.max_budget_usd = max_budget_usd
        self.timeout_seconds = timeout_seconds
        self.workspace_config = workspace_config
        self.max_context_window = max_context_window
        # Effort level for the inner Claude --effort flag. Stored on the
        # instance so the value is fixed for the life of the subprocess
        # rather than re-read from config on every claude_cmd build, and
        # so subclasses / tests can override it without monkey-patching
        # config. No _default_claude_effort_level shadow because there
        # is no workspace-level override path that would need to restore
        # a default after a /workspace switch (see workspace_config block
        # below for the pattern that DOES need a default shadow).
        self.claude_effort_level = claude_effort_level
        self.provider = "anthropic"  # Claude CLI always uses Anthropic

        # Claude-Code-specific attributes (not on the ABC)
        self.claude_user = claude_user
        self.max_session_hours = max_session_hours
        self.autocompact_pct = autocompact_pct
        self.memory_enabled = memory_enabled

        # API context for session injection (passed to build_session_context)
        self._api_context = ApiContext(
            webhook_port=webhook_port,
            webhook_secret=webhook_secret,
            services_info=services_info or [],
        )

        # Global defaults, preserved so we can restore them when
        # switching away from a configured workspace.
        self._default_model = model
        self._default_budget = max_budget_usd
        self._default_timeout = timeout_seconds

        # Apply per-workspace overrides (if configured). These become
        # the "effective" values for this workspace. The /model command
        # can still override model within a session. Model validated
        # against claude's anthropic surface so a workspaces.yaml entry
        # with a codex-only model is silently rejected here.
        if workspace_config:
            self.model = apply_workspace_model(workspace_config, "claude", "anthropic", self.model)
            if workspace_config.budget is not None:
                self.max_budget_usd = workspace_config.budget
            if workspace_config.timeout is not None:
                self.timeout_seconds = workspace_config.timeout

        self._proc: asyncio.subprocess.Process | None = None
        self._pgid: int | None = None  # Process group ID for reliable signal delivery
        self._inner_claude_pid: int | None = None  # Cross-user mode: PID of the claude grandchild under sudo (#456)
        self._effective_claude_user: str | None = None  # Cross-user mode: the resolved sudo target (#456)
        self._lock = asyncio.Lock()  # Serializes all message sends
        self._session_id: str | None = None
        self._fresh_session = True  # True until the first message is sent
        self._stderr_task: asyncio.Task | None = None  # Background stderr drain
        self._session_started_at: float | None = None  # time.monotonic() at process start

    @property
    def is_alive(self) -> bool:
        """True if the Claude subprocess is running and hasn't exited."""
        return self._proc is not None and self._proc.returncode is None

    @property
    def session_id(self) -> str | None:
        """The current Claude session ID, or None if no session is active."""
        return self._session_id

    def _session_age_hours(self) -> float:
        """Hours elapsed since the current session started."""
        if self._session_started_at is None:
            return 0.0
        return (time.monotonic() - self._session_started_at) / 3600

    def _should_recycle(self) -> bool:
        """True if the session has exceeded the configured age limit."""
        return self.max_session_hours > 0 and self.is_alive and self._session_age_hours() >= self.max_session_hours

    async def _ensure_started(self) -> None:
        """
        Start the Claude Code subprocess if not already running.

        Launches claude with stream-json I/O, bypassPermissions mode (required
        for headless operation), and the configured model and budget. The process
        runs in the current workspace directory and persists across messages.

        When claude_user is set, the process is spawned via sudo -u to run as
        a different OS user. The subprocess is started in its own process group
        (start_new_session=True) so the entire tree (sudo + claude) can be
        killed reliably via os.killpg().

        The stdout buffer limit is raised to 1 MiB (from the default 64 KiB)
        because large tool results from Claude can exceed the default.
        """
        if self.is_alive:
            return

        # Build the Claude command arguments.
        claude_cmd = [
            "claude",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            self.model,
            # Effort level controls how many reasoning tokens Claude spends
            # per turn. Passed as a CLI flag (verified against `claude --help`)
            # rather than via the --settings JSON path below, because CLI
            # flags fail loudly on typo at subprocess startup while the
            # --settings JSON path is reserved for keys that have no
            # dedicated CLI entry. Future maintainer: do NOT migrate this
            # into the --settings JSON without first re-verifying the key
            # name against `claude --help`. The value is validated at
            # config load (config.py _VALID_EFFORT_LEVELS) so by the time
            # it reaches this point it is guaranteed to be one of the
            # five accepted strings.
            "--effort",
            self.claude_effort_level,
            "--permission-mode",
            "bypassPermissions",
        ]

        # Limit context window size to reduce token usage and cache
        # invalidation pressure. Passed via --settings (not a standalone
        # CLI flag). 0 = use Claude Code's default (1M).
        if self.max_context_window > 0:
            settings = {"preferences": {"maxContextWindow": self.max_context_window}}
            claude_cmd += ["--settings", json.dumps(settings)]

        # Resolve self-sudo: skip sudo when claude_user matches the bot
        # process user. The shared utility logs a warning when skipping.
        effective_claude_user = resolve_claude_user(self.claude_user)

        if effective_claude_user:
            # -H sets HOME to the target user's pw entry. Without it, sudo
            # leaves HOME pointing at the caller, so claude reads OAuth creds
            # from the wrong ~/.claude/.credentials.json and silently exits.
            # --preserve-env=VAR (comma-separated) passes env vars through
            # sudo's env_reset. The comma-list form requires sudo >= 1.8.21
            # (2017); single-variable --preserve-env=VAR landed in 1.8.11
            # (2014). Also requires a SETENV: tag in the sudoers rule
            # (generated by install.py).
            #
            # TMPDIR is preserved so the inner claude binary writes its
            # content-hashed settings cache (claude-settings-<hex>.json)
            # into a per-os-user dir under DATA_DIR rather than into the
            # shared /tmp. The shared /tmp path otherwise causes a write
            # collision when two distinct os_users land on the same
            # hash (identical --settings JSON) - the first writer owns
            # the file at mode 0o644 and the second claude exits with
            # EACCES at startup. See issue #454.
            cmd = [
                "sudo",
                "-H",
                "-u",
                effective_claude_user,
                "--preserve-env=KAI_WEBHOOK_SECRET,TMPDIR",
                "--",
            ] + claude_cmd
        else:
            cmd = claude_cmd

        log.info(
            "Starting persistent Claude process (model=%s, user=%s)",
            self.model,
            effective_claude_user or "(same as bot)",
        )

        # Build the subprocess environment. Merge order:
        # 1. Base environment (inherited from parent process)
        # 2. Per-workspace env_file values (shared .env file)
        # 3. Per-workspace inline env values (override env_file)
        # 4. Webhook secret (LAST - workspace env can't override it)
        env = os.environ.copy()
        if self.workspace_config:
            if self.workspace_config.env_file:
                env.update(parse_env_file(self.workspace_config.env_file))
            if self.workspace_config.env:
                env.update(self.workspace_config.env)
        # Webhook secret last - ensures workspace env can't override it.
        if self._api_context.webhook_secret:
            env["KAI_WEBHOOK_SECRET"] = self._api_context.webhook_secret

        # Set autocompact threshold so Claude compacts earlier, reducing
        # token usage. Passed as an env var (not a CLI flag) because
        # Claude Code reads it from its process environment.
        #
        # The else-branch pop is load-bearing: without it, a parent
        # process that has CLAUDE_AUTOCOMPACT_PCT_OVERRIDE set (a
        # profile dotfile, /etc/kai/env loaded via load_dotenv, a
        # launchd plist EnvironmentVariables, or a manual export in
        # the shell that started the daemon) leaks the var into the
        # subprocess even when autocompact_pct=0. The contract is
        # set-or-absent on this exact key, independent of what the
        # parent env carried.
        if self.autocompact_pct > 0:
            env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] = str(self.autocompact_pct)
        else:
            env.pop("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE", None)

        # Per-os-user TMPDIR (cross-user mode only). The claude binary
        # writes its --settings JSON into `$TMPDIR/claude-settings-<hex>
        # .json` where <hex> is content-hashed from the settings string.
        # When two distinct os_users spawn claude with the same
        # --settings JSON, both land on the same default-/tmp path; the
        # first writer owns the file at mode 0o644 and the second
        # claude exits with EACCES on the write-open. Anchoring TMPDIR
        # to `<DATA_DIR>/tmp/<os_user>/` (per-user dir created and
        # chowned by install.py `_apply_migrate`) gives each os_user
        # its own temp namespace. Single-user mode (no claude_user)
        # uses the system default /tmp; no cross-user collision is
        # possible there. See issue #454.
        if effective_claude_user:
            env["TMPDIR"] = str(DATA_DIR / "tmp" / effective_claude_user)

        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.workspace),
            env=env,
            limit=1024 * 1024,  # 1 MiB; default 64 KiB too small for large tool results
            # When spawned via sudo, start in a new process group so we can
            # kill the entire tree (sudo + claude) via os.killpg(). Without
            # this, killing sudo may orphan the claude process.
            start_new_session=bool(effective_claude_user),
        )
        self._session_id = None
        self._fresh_session = True
        self._session_started_at = time.monotonic()

        # Save the process group ID for reliable signal delivery.
        # When claude_user is set, start_new_session=True creates a new group
        # with PGID == PID (session leader). Save it now because os.getpgid()
        # fails after the process exits, but os.killpg() works as long as any
        # group member is still alive (i.e., the actual claude process).
        if effective_claude_user:
            self._pgid = self._proc.pid  # PGID == PID for session leaders
        else:
            self._pgid = None

        # Remember the resolved sudo target for cross-user signal escalation
        # in _send_signal / _kill (#456). os.killpg from the service user
        # cannot signal a target-user claude grandchild (POSIX EPERM); we
        # need to escalate via `sudo -n -u <target> kill ...` instead, which
        # requires us to know who to sudo to at signal time. _inner_claude_pid
        # is reset to None here so a recycled instance does not carry a
        # stale PID from a previous spawn into a fresh process tree.
        self._effective_claude_user = effective_claude_user
        self._inner_claude_pid = None

        # Drain stderr in background to prevent pipe buffer deadlock
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def _drain_stderr(self) -> None:
        """
        Continuously read and discard stderr from the Claude process.

        Without this, the stderr pipe buffer fills up and the process deadlocks.
        Lines are logged at DEBUG level (truncated to 200 chars) for diagnostics.
        """
        while self._proc and self._proc.stderr:
            try:
                line = await self._proc.stderr.readline()
                if not line:
                    break
                text = line.decode().strip()
                if text:
                    log.debug("Claude stderr: %s", text[:200])
            except Exception:
                log.warning("Unexpected error in stderr drain", exc_info=True)
                break

    def _lookup_inner_claude_pid(self) -> int | None:
        """
        Find the inner claude subprocess PID in cross-user mode (#456).

        In cross-user mode the bot spawns `sudo -u <target> -- claude ...`
        and self._proc tracks the sudo wrapper. sudo fork+exec's claude
        as its sole child, so pgrep -P <sudo_pid> returns the claude
        PID. Returns None when:
        - no sudo wrapper is alive (single-user mode, or sudo already
          reaped),
        - sudo has not yet forked (very early in the spawn lifecycle,
          before claude exists), or
        - pgrep is not available / errors out.

        The result is cached on self._inner_claude_pid; callers that
        need to signal after sudo has been killed must rely on the
        cache (pgrep returns nothing on a reaped sudo). _kill primes
        the cache before its first signal for this reason.

        Synchronous variant used by the sync `_send_signal` path
        (which is called from `force_kill`, also sync). The async
        sites (`_kill`, `shutdown`) call `_async_lookup_inner_claude_pid`
        instead so they do not block the event loop on the pgrep
        invocation. See #459.
        """
        if self._proc is None:
            return None
        try:
            result = subprocess.run(
                ["pgrep", "-P", str(self._proc.pid)],
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

    async def _async_lookup_inner_claude_pid(self) -> int | None:
        """
        Async pgrep equivalent of `_lookup_inner_claude_pid` (#459).

        Same semantics as the synchronous version - returns the inner
        claude PID by querying pgrep against the sudo wrapper's PID -
        but uses `asyncio.create_subprocess_exec` so the event loop
        is not blocked while pgrep runs. Called from `_kill` and
        `shutdown`; the sync variant remains for the `force_kill`
        path. The 2s ceiling matches the sync version's timeout.
        """
        if self._proc is None:
            return None
        try:
            proc = await asyncio.create_subprocess_exec(
                "pgrep",
                "-P",
                str(self._proc.pid),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except (OSError, FileNotFoundError):
            return None
        try:
            stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=2)
        except TimeoutError:
            # pgrep is hung; reap it so we do not leak a child and
            # return None as if no PID was found.
            #
            # Guard proc.kill() against the race where pgrep exits
            # between wait_for timing out and the kill() call:
            # asyncio.subprocess.Process.kill() delegates to
            # os.kill(pid, SIGKILL), which raises
            # ProcessLookupError (an OSError subclass) on an
            # already-dead PID. Swallow because the outcome we
            # wanted (process gone) is already true.
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

    async def _async_send_signal_for_close(self, sig: int) -> None:
        """
        Async equivalent of `_send_signal` used by `_kill` and
        `shutdown` (#459).

        Same three-step structure as the sync version:
        1. Cross-user mode: prime the inner-claude-PID cache (async
           pgrep) and signal the inner claude through `sudo -n -u
           <target> /bin/kill` (async subprocess).
        2. Wrapper reap via `os.killpg` (non-blocking syscall, sync
           is fine).
        3. Single-user mode: `_proc.send_signal` (also non-blocking).

        Step 1's blocking surface is moved off the event loop; the
        sync `_send_signal` remains for `force_kill` (which is sync
        by public-API contract).
        """
        if self._pgid is not None:
            if self._effective_claude_user is not None:
                if self._inner_claude_pid is None:
                    self._inner_claude_pid = await self._async_lookup_inner_claude_pid()
                if self._inner_claude_pid is not None:
                    await self._async_sudo_kill(
                        self._effective_claude_user,
                        self._inner_claude_pid,
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

    async def _async_sudo_kill(self, target_user: str, pid: int, sig: int) -> None:
        """
        Run `sudo -n -u <target> /bin/kill -<sig> <pid>` without blocking
        the event loop (#459).

        Equivalent to the synchronous `subprocess.run` call inside
        `_send_signal` and the final-cleanup blocks of `_kill`/
        `shutdown`, but uses `asyncio.create_subprocess_exec` so a
        long-running sudo or kill (system pathology) does not stall
        other coroutines. The 5s ceiling matches the sync ceiling.

        Logs at WARNING when sudo returns non-zero so a missing
        sudoers rule, an ESRCH on a double-kill, or any other
        failure mode surfaces in the operator-facing log instead
        of silently leaking an orphan (same diagnostic contract
        the sync path adopts in #458/M1).

        Silent on TimeoutError and OSError: same failure-mode
        swallow as the sync path. The caller has no remedy for
        a hung sudo or a missing /bin/kill; logging the timeout
        once at WARNING is the best we can do.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "sudo",
                "-n",
                "-u",
                target_user,
                "/bin/kill",
                f"-{int(sig)}",
                str(pid),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (OSError, FileNotFoundError):
            return
        try:
            _, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=5)
        except TimeoutError:
            # Same race guard as _async_lookup_inner_claude_pid:
            # proc.kill() can raise ProcessLookupError if sudo
            # exited between wait_for timing out and the kill()
            # call. Swallow because the outcome we wanted
            # (process gone) is already true.
            try:
                proc.kill()
            except OSError:
                pass
            try:
                await proc.wait()
            except OSError:
                pass
            log.warning(
                "async sudo kill timed out after 5s (target=%s, pid=%d, sig=%d); inner claude may orphan",
                target_user,
                pid,
                int(sig),
            )
            return
        if proc.returncode != 0:
            log.warning(
                "sudo kill escalation failed (rc=%d, stderr=%r); inner claude pid=%d may orphan",
                proc.returncode,
                stderr_bytes[:200].decode(errors="replace") if stderr_bytes else "",
                pid,
            )

    def _send_signal(self, sig: int) -> None:
        """
        Send a signal to the Claude process (or process group if claude_user).

        Deliberately does NOT check self._proc.returncode. When claude_user
        is set, self._proc tracks the sudo wrapper, not the actual claude
        process. If sudo exits before claude (e.g., from SIGTERM), checking
        returncode would skip sending further signals - leaving claude
        orphaned. Instead, we always attempt delivery and let OSError handle
        the already-dead case cleanly:
        - claude_user path: os.killpg() raises OSError if the group is gone
        - direct path: send_signal() calls os.kill() which raises OSError

        Cross-user mode (#456): os.killpg from the service user cannot
        signal a target-user claude grandchild (POSIX signal permission
        rules: real or effective UID must match real or saved set-user-ID
        of the target). killpg succeeds for the sudo wrapper (real UID is
        the service user) but the inner claude (real, effective, and
        saved UIDs = target) survives and gets orphaned to init. To
        avoid this, we escalate via `sudo -n -u <target> kill -<sig>
        <pid>` for the inner claude BEFORE killpg reaps the sudo
        wrapper. install.py's `_generate_sudoers` emits a per-target
        `NOPASSWD: /bin/kill` rule for this path.

        Args:
            sig: Signal to send (e.g., signal.SIGTERM, signal.SIGKILL).
        """
        if self._pgid is not None:
            # Cross-user mode: signal the inner claude through sudo
            # escalation FIRST so killpg's reap of the sudo wrapper
            # does not orphan it. Only attempt the lookup when we
            # actually have a sudo target to escalate to; without
            # _effective_claude_user there is no target to pass to
            # `sudo -u` and pgrep would be a wasted call.
            if self._effective_claude_user is not None:
                if self._inner_claude_pid is None:
                    self._inner_claude_pid = self._lookup_inner_claude_pid()
                if self._inner_claude_pid is not None:
                    try:
                        # Absolute /bin/kill path so sudo's
                        # secure_path resolution cannot silently
                        # pick a different binary than the sudoers
                        # rule names. The rule generated by
                        # install.py also pins /bin/kill; the two
                        # must match exactly. See #458 review.
                        #
                        # int(sig) explicit because IntEnum
                        # __format__ behavior changed in 3.11: on
                        # Python < 3.11 an f-string of signal.SIGKILL
                        # produces "Signals.SIGKILL" rather than "9",
                        # which /bin/kill would reject silently
                        # (check=False) and recreate the orphan-leak.
                        # Production runs 3.13 so the f-string would
                        # do the right thing here, but the int()
                        # cast is the contract regardless of version.
                        result = subprocess.run(
                            [
                                "sudo",
                                "-n",
                                "-u",
                                self._effective_claude_user,
                                "/bin/kill",
                                f"-{int(sig)}",
                                str(self._inner_claude_pid),
                            ],
                            capture_output=True,
                            timeout=5,
                            check=False,
                        )
                        # Surface non-zero sudo exits at WARNING.
                        # The original orphan-leak (#456) was
                        # invisible at INFO because the inherited
                        # OSError swallow had no log line; the
                        # check=False + silent return on a missing
                        # NOPASSWD rule would recreate the same
                        # blind spot here. ESRCH from a kill on
                        # an already-dead PID is the most common
                        # benign nonzero, but it is uncommon
                        # enough on a healthy spawn that flagging
                        # it is cheaper than the next "we did not
                        # know it was orphaning" forensic round.
                        if result.returncode != 0:
                            log.warning(
                                "sudo kill escalation failed (rc=%d, stderr=%r); inner claude may orphan",
                                result.returncode,
                                result.stderr[:200].decode(errors="replace") if result.stderr else "",
                            )
                    except (subprocess.TimeoutExpired, OSError):
                        # Inner claude already dead, sudo not allowed
                        # (sudoers rule missing on an old install), or
                        # the kill binary times out. Fall through to
                        # killpg below - we did our best on the inner
                        # process; the wrapper still needs reaping.
                        pass
            # Wrapper cleanup: signal the entire process group so
            # the sudo parent is reaped regardless of whether the
            # inner claude is signal-deliverable from the service
            # user. Same OSError-swallow pattern as before.
            try:
                os.killpg(self._pgid, sig)
            except OSError:
                pass
        elif self._proc is not None:
            try:
                self._proc.send_signal(sig)
            except OSError:
                pass

    async def send(self, prompt: str | list, chat_id: int | None = None) -> AsyncIterator[StreamEvent]:
        """
        Send a message to Claude and yield streaming events.

        This is the main public interface. All sends are serialized via an
        internal lock so concurrent callers (e.g., a user message arriving
        while a cron job is running) queue rather than interleave.

        Args:
            prompt: Either a text string or a list of content blocks (for
                multi-modal messages like images).
            chat_id: Optional Telegram chat ID of the user. When provided
                on the first message of a session, it's included in the
                context so inner Claude can route API calls to the correct
                user. Forward-compatible with Phase 3 per-user subprocesses.

        Yields:
            StreamEvent objects with accumulated text. The final event has
            done=True and includes the complete AgentResponse.
        """
        async with self._lock:
            async for event in self._send_locked(prompt, chat_id=chat_id):
                yield event

    async def _send_locked(self, prompt: str | list, chat_id: int | None = None) -> AsyncIterator[StreamEvent]:
        """
        Core message-sending logic (must be called while holding self._lock).

        Handles the full lifecycle of a single Claude interaction:
        1. Ensure the subprocess is alive (start if needed)
        2. On the first message of a new session, prepend identity, memory,
           conversation history, and scheduling API context to the prompt
        3. When in a foreign workspace, prepend a per-message reminder to
           prevent Claude from acting on workspace context autonomously
        4. Write the JSON message to stdin and stream stdout line-by-line
        5. Parse stream-json events and yield StreamEvents to the caller

        Args:
            prompt: Either a text string or a list of content blocks (for
                multi-modal messages like images).

        Yields:
            StreamEvent objects with accumulated text. The final event has
            done=True and includes the complete AgentResponse.
        """
        # Recycle the session if it has exceeded the age limit. This prevents
        # unbounded memory growth in the inner Claude process (Node.js/V8),
        # which can cause macOS kernel panics via Jetsam on memory-constrained
        # machines. Only checked before starting a new interaction, never
        # during one, so in-flight responses complete normally.
        if self._should_recycle():
            log.info(
                "Session age %.1f hours exceeds limit of %.1f hours; recycling",
                self._session_age_hours(),
                self.max_session_hours,
            )
            # Shorter timeout for recycle since the user is waiting
            await self._save_prompt(timeout=10)
            await self._kill()

        try:
            await self._ensure_started()
        except FileNotFoundError:
            yield StreamEvent(
                text_so_far="",
                done=True,
                response=AgentResponse(success=False, text="", error="claude CLI not found"),
            )
            return

        # Per-turn prompt context. Build the fresh-session bootstrap
        # (MEMORY.md / PREFERENCES.md seed + session_context block)
        # and the foreign-workspace reminder LOCALLY, then hand both
        # to `assemble_turn_context` so the shared helper owns the
        # ordering invariant (marker closest to user text, session/
        # memory/reminder stacked above in that order).
        #
        # Fresh-session lifecycle stays in the backend because the
        # `_fresh_session` flag, the MEMORY.md/PREFERENCES.md seeding
        # (and its swallowed-OSError contract), and the
        # `build_session_context` inputs (workspace, home_workspace,
        # API context, workspace_config) are all backend-owned state.
        # The shared helper handles only string composition; pulling
        # the lifecycle into it would couple the helper to per-
        # backend internals it has no other reason to know about.
        #
        # The single-call MEMORY.md/PREFERENCES.md bootstrap retry
        # caveat from before extraction still applies: the seed runs
        # exactly once per session because `_fresh_session` flips to
        # False unconditionally; a transient OSError inside
        # `ensure_user_memory` / `ensure_user_preferences` is logged
        # as a warning but not retried on subsequent messages of the
        # same session. Failure modes here are persistent
        # (permissions, missing parent dir), not transient, so the
        # one-shot behavior is acceptable; documenting it so future
        # readers do not assume self-healing.
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
        # can change between messages via `/workspace`). `build_
        # foreign_workspace_reminder` returns None when the workspace
        # equals home_workspace; `assemble_turn_context` no-ops on an
        # empty string, so coerce the None to "" rather than threading
        # the optional through the helper signature.
        reminder = build_foreign_workspace_reminder(self.workspace, self.home_workspace) or ""

        prompt = await assemble_turn_context(
            prompt,
            chat_id=chat_id,
            session_context=session_ctx,
            workspace_reminder=reminder,
        )

        content = prompt if isinstance(prompt, list) else [{"type": "text", "text": prompt}]
        msg = (
            json.dumps(
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": content,
                    },
                }
            )
            + "\n"
        )

        assert self._proc is not None
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None
        try:
            self._proc.stdin.write(msg.encode())
            await self._proc.stdin.drain()
        except OSError as e:
            log.error("Failed to write to Claude process: %s", e)
            await self._kill()
            yield StreamEvent(
                text_so_far="",
                done=True,
                response=AgentResponse(success=False, text="", error="Claude process died, restarting on next message"),
            )
            return

        accumulated_text = ""
        # Idle-activity timeout for the interaction. Resets every time the
        # process emits a line of output. If the process goes silent for
        # this long, it is likely dead or wedged. The per-readline timeout
        # below (timeout_seconds * 3) is the primary dead-process detector;
        # this is a secondary safety net measured across the whole interaction.
        last_activity = time.monotonic()
        max_idle_seconds = self.timeout_seconds * 5  # 10 min of silence at default 120s
        try:
            while True:
                # Check idle timeout before each readline
                idle_seconds = time.monotonic() - last_activity
                if idle_seconds > max_idle_seconds:
                    log.error(
                        "Interaction idle timeout (%.0fs with no output, limit %ds)",
                        idle_seconds,
                        max_idle_seconds,
                    )
                    await self._kill()
                    yield StreamEvent(
                        text_so_far=accumulated_text,
                        done=True,
                        response=AgentResponse(
                            success=False,
                            text=accumulated_text,
                            error="Claude interaction timed out (no output)",
                        ),
                    )
                    return

                try:
                    # Opus with tool use can go minutes between output lines
                    timeout = self.timeout_seconds * 3
                    line = await asyncio.wait_for(self._proc.stdout.readline(), timeout=timeout)
                except TimeoutError:
                    log.error("Claude response timed out")
                    await self._kill()
                    yield StreamEvent(
                        text_so_far=accumulated_text,
                        done=True,
                        response=AgentResponse(success=False, text=accumulated_text, error="Claude timed out"),
                    )
                    return

                # Reset idle timer - process is alive and producing output.
                # Do NOT reset on empty line (EOF); that means the process died.
                if line:
                    last_activity = time.monotonic()

                if not line:
                    # Process died unexpectedly
                    log.error("Claude process EOF")
                    await self._kill()
                    yield StreamEvent(
                        text_so_far=accumulated_text,
                        done=True,
                        response=AgentResponse(
                            success=bool(accumulated_text),
                            text=accumulated_text,
                            error=None if accumulated_text else "Claude process ended unexpectedly",
                        ),
                    )
                    return

                try:
                    event = json.loads(line.decode())
                except json.JSONDecodeError:
                    log.debug("Skipping non-JSON stdout line: %s", line.decode().strip()[:200])
                    continue

                etype = event.get("type")

                if etype == "system":
                    sid = event.get("session_id")
                    if sid:
                        self._session_id = sid

                elif etype == "result":
                    # Prefer accumulated_text (which includes text before tool
                    # use) over the result event's text (which may only contain
                    # the final assistant message). Fall back to result_text
                    # when nothing was accumulated (e.g., system-only responses).
                    result_text = event.get("result", "")
                    text = accumulated_text if accumulated_text else result_text
                    # When the CLI signals an error but does not populate the
                    # result field, the user sees the literal string "Error:
                    # None" with no clue about the error class. Log the known
                    # safe metadata fields individually so diagnosis does not
                    # depend on assumptions about the CLI's event schema: the
                    # event could in principle gain new fields that contain
                    # model output or user content, and logging the full dict
                    # would silently leak them. Log `sorted(event.keys())`
                    # rather than the full dict so that a schema addition is
                    # visible (key name only, not value), and a follow-up can
                    # opt-in to logging the new field once its shape is known.
                    # See issue #326 for the paired UX fix.
                    if event.get("is_error", False) and not result_text:
                        log.warning(
                            "Result event with is_error=true has no result field; "
                            "session=%s cost_usd=%s duration_ms=%s keys=%s",
                            event.get("session_id"),
                            event.get("total_cost_usd"),
                            event.get("duration_ms"),
                            sorted(event.keys()),
                        )
                    # Resolve the error string for downstream consumers.
                    # The CLI's `is_error=true` events come in two shapes:
                    #
                    #   (a) `result` populated with a human-readable
                    #       reason. Use it directly.
                    #   (b) `result` empty BUT `errors` populated with a
                    #       list of strings (the documented variant for
                    #       BUDGET_CEILING exhaustion: errors carries
                    #       ["Reached maximum budget ($N)"]).
                    #
                    # Falls back to a non-None sentinel when both fields
                    # are empty so downstream rendering never produces
                    # the literal "Error: None" string in chat, even on
                    # a future CLI variant that emits is_error=true with
                    # neither field populated.
                    #
                    # Note on `result_text`: it is `event.get("result", "")`
                    # captured a few lines above, with no transformation
                    # applied. Using `result_text` here (rather than
                    # `event.get("result")` again) is equivalent and avoids
                    # the second dict lookup; the truthiness check still
                    # catches the empty-string case which is what the
                    # branch table targets.
                    response_error: str | None = None
                    if event.get("is_error", False):
                        if result_text:
                            response_error = result_text
                        else:
                            errors_list = event.get("errors")
                            if isinstance(errors_list, list) and errors_list:
                                response_error = "; ".join(str(e) for e in errors_list)
                            else:
                                response_error = "no error detail provided"
                    response = AgentResponse(
                        success=not event.get("is_error", False),
                        text=text,
                        session_id=event.get("session_id", self._session_id),
                        cost_usd=event.get("total_cost_usd", 0.0),
                        duration_ms=event.get("duration_ms", 0),
                        error=response_error,
                    )
                    yield StreamEvent(text_so_far=response.text, done=True, response=response)
                    return

                elif etype == "assistant" and "message" in event:
                    msg_data = event["message"]
                    if isinstance(msg_data, dict) and "content" in msg_data:
                        for block in msg_data["content"]:
                            if block.get("type") == "text":
                                new_text = block.get("text", "")
                                if accumulated_text and new_text and not accumulated_text.endswith("\n"):
                                    accumulated_text += "\n\n"
                                accumulated_text += new_text
                                yield StreamEvent(text_so_far=accumulated_text)

        except Exception as e:
            log.exception("Unexpected error reading Claude stream")
            await self._kill()
            yield StreamEvent(
                text_so_far=accumulated_text,
                done=True,
                response=AgentResponse(success=False, text=accumulated_text, error=str(e)),
            )

    def force_kill(self) -> None:
        """
        Kill the subprocess immediately. Safe to call without holding the lock.

        Called by /stop to abort an in-flight response. There is a race window
        between _ensure_started() and the stdin write in _send_locked(), but
        it is safe: killing the process causes EOF on stdout, which the
        streaming loop handles by yielding a done event and calling _kill()
        to clean up. No lock acquisition is needed here because _send_signal()
        only sends a signal and is itself idempotent.
        """
        self._send_signal(signal.SIGKILL)
        # Cancel the stderr drain so it does not outlive the process.
        # For /stop, _kill() will see _stderr_task=None and skip its own
        # cancel. For eviction, this is the only cleanup point.
        if self._stderr_task:
            self._stderr_task.cancel()
            self._stderr_task = None

    async def _save_prompt(self, timeout: float = 30) -> None:
        """
        Send a save prompt to the inner Claude before shutdown.

        Gives the subprocess a chance to persist useful context to MEMORY.md
        before it is killed. Best-effort: if the prompt times out or the
        process is unresponsive, the caller proceeds with shutdown anyway.

        Args:
            timeout: Maximum seconds to wait for the save operation.
                Defaults to 30s for idle eviction/graceful shutdown.
                Use a shorter value (e.g., 10s) for user-visible paths
                like session recycle where latency matters.

        Prerequisites:
          - self._proc is alive and responsive
          - No other interaction is in flight (caller holds exclusive pipe access)
          - self._fresh_session is False (at least one real message was processed)
        """
        if self._fresh_session or not self.is_alive:
            return

        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            return

        save_msg = (
            "You are about to be shut down. Save anything worth remembering "
            "from this session to your memory file. Only save genuinely "
            "useful information - user preferences, personal facts, decisions, "
            "corrections, or important context. Do not save session-specific "
            "details like current task progress or temporary debugging state."
        )

        content = [{"type": "text", "text": save_msg}]
        msg = (
            json.dumps(
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": content,
                    },
                }
            )
            + "\n"
        )

        try:
            self._proc.stdin.write(msg.encode())
            await self._proc.stdin.drain()
        except (OSError, RuntimeError):
            # OSError: pipe broken. RuntimeError: transport closed during
            # drain(). Either way, the process is dying. Nothing to save.
            log.debug("Save prompt write failed; process already dying")
            return

        # Read and discard response lines until we see the result event or
        # timeout. The timeout covers the entire save operation (read current
        # memory, decide what to save, write file, respond).
        try:
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    log.warning("Save prompt timed out; proceeding with shutdown")
                    break
                try:
                    line = await asyncio.wait_for(self._proc.stdout.readline(), timeout=remaining)
                except TimeoutError:
                    log.warning("Save prompt timed out; proceeding with shutdown")
                    break
                if not line:
                    # EOF - process died during save
                    break
                # Parse the line to check for completion. The inner Claude
                # responds via stream-json, so look for "result" type events.
                try:
                    event = json.loads(line)
                    if event.get("type") == "result":
                        # Final event - save interaction complete
                        log.info("Save prompt completed successfully")
                        break
                except (json.JSONDecodeError, ValueError):
                    # Non-JSON line (stderr bleed, etc.) - skip
                    continue
        except Exception:
            # Any error reading the response - log and move on.
            # The subprocess will be killed momentarily regardless.
            log.debug("Save prompt response read failed; proceeding with shutdown")

    async def change_workspace(
        self,
        new_workspace: Path,
        workspace_config: WorkspaceConfig | None = None,
    ) -> None:
        """
        Switch to a new workspace directory and apply its config.

        Kills the current subprocess. The next send() restarts Claude
        in the new directory with the new config applied.

        Args:
            new_workspace: Path to the new working directory.
            workspace_config: Per-workspace config for the target, or
                None to use global defaults.
        """
        # Kill first, then mutate. An in-flight _send_locked() reads
        # self.workspace, self.timeout_seconds, and self.workspace_config
        # at various await points during streaming. If we mutate before
        # killing, the stream sees new config values while still running
        # the old workspace's process. Killing first ensures the stream
        # hits EOF and exits before any state changes.
        await self._kill()

        self.workspace = new_workspace
        self.workspace_config = workspace_config

        # Always revert to global defaults first, then apply overrides.
        # This prevents stale values when switching from a fully-configured
        # workspace to a partially-configured one (e.g., workspace A has
        # budget=15.0 but workspace B only sets model - without the reset,
        # budget would carry over from A instead of reverting to default).
        self.model = self._default_model
        self.max_budget_usd = self._default_budget
        self.timeout_seconds = self._default_timeout

        if workspace_config:
            self.model = apply_workspace_model(workspace_config, "claude", "anthropic", self.model)
            if workspace_config.budget is not None:
                self.max_budget_usd = workspace_config.budget
            if workspace_config.timeout is not None:
                self.timeout_seconds = workspace_config.timeout

    async def restart(self) -> None:
        """
        Kill the current process so the next send() starts fresh.
        Called by /new command and model switches.
        """
        await self._kill()

    async def _kill(self) -> None:
        """
        Kill the subprocess immediately and clean up resources.

        Sends SIGKILL, waits up to 5 seconds for exit, then clears all
        process state. After clearing, sends one final SIGKILL to the
        saved process group to catch any orphaned children that survived
        the initial signal (e.g., claude reparented to init after sudo
        died). The timeout prevents hanging on zombie processes.
        Idempotent - safe to call even if the process has already exited.

        Note: _stderr_task cancellation is inside the `if self._proc` guard
        because _stderr_task is only created alongside _proc in _ensure_started().
        If _proc is None, there is no stderr task to cancel.
        """
        if self._proc:
            # Save pgid before clearing - the EOF handler in _send_locked()
            # may call _kill() again after we clear self._pgid, but we need
            # to ensure the process group gets signaled at least once more
            # after the wait completes (belt-and-suspenders for the race
            # where sudo dies but claude survives the initial SIGKILL).
            saved_pgid = self._pgid

            # Prime the inner-claude-PID cache BEFORE any signal goes
            # out. Once the killpg below reaps the sudo wrapper,
            # pgrep -P <dead_sudo_pid> returns nothing and we lose
            # the only handle on the orphaned grandchild. The cache
            # survives the per-signal escalation + killpg sequence
            # so the final-cleanup sudo-kill below can still reach
            # the inner claude. See issue #456. Only attempt when
            # we actually have a sudo target to escalate to; without
            # _effective_claude_user the lookup result is useless.
            #
            # Async pgrep variant so the event loop is not blocked
            # waiting for the lookup. See #459.
            saved_user = self._effective_claude_user
            if saved_user is not None and self._inner_claude_pid is None:
                self._inner_claude_pid = await self._async_lookup_inner_claude_pid()
            saved_inner_pid = self._inner_claude_pid

            # Cross-user escalation (#456) + wrapper killpg / single-
            # user direct signal. Routed through the async helper so
            # the event loop stays responsive while sudo + kill
            # complete. See #459.
            await self._async_send_signal_for_close(signal.SIGKILL)

            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except TimeoutError:
                pass

            # Cancel the stderr drain BEFORE clearing self._proc.
            # _drain_stderr's while-loop checks self._proc on each iteration;
            # if we clear proc first, the drain task could observe None in a
            # state that was never intended to be visible to it. Cancelling
            # the task first ensures it stops reading before its dependencies
            # are destroyed.
            if self._stderr_task:
                self._stderr_task.cancel()
                self._stderr_task = None

            self._proc = None
            self._pgid = None
            self._inner_claude_pid = None
            self._effective_claude_user = None
            self._session_id = None
            self._session_started_at = None

            # Final cleanup: signal the saved process group one more time.
            # If claude was reparented to init during the wait, this catches
            # it. If everything is already dead, killpg raises OSError which
            # we ignore. Only applies to claude_user mode (pgid is None
            # otherwise).
            if saved_pgid is not None:
                try:
                    os.killpg(saved_pgid, signal.SIGKILL)
                except OSError:
                    pass

            # Final cross-user cleanup: even after the killpg above,
            # the inner claude grandchild may still be alive because
            # the service user cannot signal a target-user process
            # (#456). Escalate via sudo to kill the captured PID
            # directly. Skipped silently when:
            # - single-user mode (no saved_user / saved_inner_pid),
            # - pgrep failed to find a child PID at any point
            #   (sudo never forked, or sudo died with no child),
            # - the inner claude is already dead (sudo kill returns
            #   ESRCH; sudo exits 1; logged at WARNING by
            #   _async_sudo_kill).
            #
            # Async sudo-kill so the event loop is not blocked on the
            # 5s ceiling. See #459.
            if saved_user is not None and saved_inner_pid is not None:
                await self._async_sudo_kill(saved_user, saved_inner_pid, int(signal.SIGKILL))

    async def shutdown(self) -> None:
        """
        Gracefully shut down the Claude process.

        Acquires the send lock for the entire sequence so that
        _save_prompt() (which reads stdout) cannot overlap with a new
        send() call's readline(). If a stream is in progress, we wait
        for it to finish before proceeding.

        Sends SIGTERM first and waits up to 5 seconds for clean exit.
        Falls back to SIGKILL if the process doesn't terminate in time.

        Unlike the old implementation, this does NOT check returncode
        before sending signals. When claude_user is set, self._proc
        tracks sudo, not claude - if sudo exits from SIGTERM before
        claude does, the returncode guard would skip the SIGKILL
        fallback, orphaning claude. _send_signal() handles
        already-dead processes via OSError instead.
        """
        # Hold the lock through save + terminate so no send() can
        # start a concurrent stdout read during _save_prompt().
        # If a stream is in flight, this blocks until it finishes.
        # Note: _kill() (called from _send_locked error paths) does
        # NOT acquire _lock, so there is no deadlock risk.
        saved_pgid = None
        saved_inner_pid: int | None = None
        saved_user: str | None = None
        async with self._lock:
            if self._proc:
                try:
                    await self._save_prompt()
                except Exception:
                    log.warning("save_prompt failed during shutdown; proceeding with termination", exc_info=True)

                saved_pgid = self._pgid

                # Prime the inner-claude-PID cache BEFORE the SIGTERM
                # so the post-killpg cleanup below can still escalate
                # via sudo even after the wrapper is reaped. Same
                # rationale as _kill; see issue #456. Gated on
                # _effective_claude_user because without a sudo
                # target the lookup result is useless.
                #
                # Async pgrep variant so the event loop stays
                # responsive while pgrep runs. See #459.
                saved_user = self._effective_claude_user
                if saved_user is not None and self._inner_claude_pid is None:
                    self._inner_claude_pid = await self._async_lookup_inner_claude_pid()
                saved_inner_pid = self._inner_claude_pid

                # Per-signal cross-user escalation runs ASYNC here so
                # the event loop is not stalled while sudo + kill
                # complete. killpg is a non-blocking syscall and
                # stays sync; single-user fallback uses _proc.
                # send_signal (also non-blocking). See #459.
                await self._async_send_signal_for_close(signal.SIGTERM)
                try:
                    await asyncio.wait_for(self._proc.wait(), timeout=5)
                except TimeoutError:
                    await self._async_send_signal_for_close(signal.SIGKILL)
                    try:
                        await asyncio.wait_for(self._proc.wait(), timeout=5)
                    except TimeoutError:
                        log.warning("Process did not exit after SIGKILL; abandoning")

            # Cancel stderr drain before clearing proc (same invariant
            # as _kill: the drain task checks self._proc, so cancel it
            # first).
            if self._stderr_task:
                self._stderr_task.cancel()
                self._stderr_task = None

            # Clean up state regardless of how (or whether) the process
            # exited. Done inside the lock so any queued send() sees
            # _proc=None and starts a fresh process via _ensure_started.
            self._proc = None
            self._pgid = None
            self._inner_claude_pid = None
            self._effective_claude_user = None
            self._session_started_at = None

        # Final cleanup: signal the saved process group one more time
        # to catch any orphaned children that survived the initial
        # signals. Done outside the lock since it touches no shared
        # reader state.
        if saved_pgid is not None:
            try:
                os.killpg(saved_pgid, signal.SIGKILL)
            except OSError:
                pass

        # Final cross-user cleanup: signal the captured inner claude
        # PID via sudo escalation. See _kill for the full rationale;
        # in summary, the service user cannot signal a target-user
        # process directly, so killpg above can leave the inner
        # claude orphaned to init. Skipped silently when the spawn
        # was not in cross-user mode or pgrep never found a child.
        #
        # Async sudo-kill so the event loop is not blocked on the
        # 5s ceiling. See #459.
        if saved_user is not None and saved_inner_pid is not None:
            await self._async_sudo_kill(saved_user, saved_inner_pid, int(signal.SIGKILL))
