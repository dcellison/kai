"""
Agent backend abstraction and shared context injection.

Provides:
1. AgentBackend ABC - the interface that pool.py programs against
2. AgentResponse / StreamEvent - backend-agnostic protocol types
3. ApiContext - groups webhook/service info for context injection
4. Context injection functions - build the identity/memory/history/API
   prefix that gets prepended to the first message of each session

The ABC defines the minimal surface that SubprocessPool needs. Concrete
backends (ClaudeCodeBackend, future GooseBackend, etc.) implement the
process management; the context functions here handle the Kai-specific
prompt assembly that is identical across all backends.
"""

import logging
import os
import shutil
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

from kai.config import DATA_DIR, PROJECT_ROOT, Config, WorkspaceConfig
from kai.history import get_recent_history

log = logging.getLogger(__name__)


# ── Protocol types ──────────────────────────────────────────────────


@dataclass
class AgentResponse:
    """
    Final response from an agent backend interaction.

    Attributes:
        success: True if the backend returned a valid response, False on error.
        text: The full response text (accumulated from streaming chunks).
        session_id: Session identifier (used for session continuity).
        cost_usd: Cost of this interaction in USD.
        duration_ms: Wall-clock duration of the interaction in milliseconds.
        error: Error message if success is False, None otherwise.
    """

    success: bool
    text: str
    session_id: str | None = None
    cost_usd: float = 0.0
    duration_ms: int = 0
    error: str | None = None


@dataclass
class StreamEvent:
    """
    A partial update emitted during a backend's streaming response.

    Yielded by AgentBackend.send() as the backend generates text. The
    final event has done=True and includes the complete AgentResponse.

    Attributes:
        text_so_far: Accumulated response text up to this point.
        done: True if this is the final event (response complete or error).
        response: The complete AgentResponse, set only when done=True.
    """

    text_so_far: str
    done: bool = False
    response: AgentResponse | None = None


# ── API context ─────────────────────────────────────────────────────


@dataclass
class ApiContext:
    """
    Webhook and service proxy info for context injection.

    Groups the three API-related values that build_session_context()
    needs, keeping the function signature manageable.

    Attributes:
        webhook_port: Local port the webhook server listens on.
        webhook_secret: Shared secret for authenticating API requests.
        services_info: List of dicts describing available external services.
    """

    webhook_port: int
    webhook_secret: str
    services_info: list[dict] = field(default_factory=list)


# ── Abstract backend ────────────────────────────────────────────────


class AgentBackend(ABC):
    """
    Abstract base for agent subprocess backends.

    Backends manage a persistent subprocess that accepts prompts and
    streams responses. The pool (pool.py) owns backend instances and
    calls these methods; it does not care which agent harness is behind
    them.

    Mutable attributes on the ABC (set by pool.py during workspace
    restore, /settings, and /model commands). These are generic to any
    backend:
        model, workspace, home_workspace, max_budget_usd,
        timeout_seconds, max_context_window, workspace_config, provider

    Backend-specific attributes (NOT on the ABC) stay on the concrete
    class. For ClaudeCodeBackend these include: claude_user,
    max_session_hours, autocompact_pct. The pool never touches these
    directly; they are set at construction by _create_instance().
    """

    # These are plain instance attributes, not abstract properties,
    # because pool.py reads AND writes them directly (e.g.,
    # instance.model = "opus"). Abstract properties with setters are
    # verbose for no benefit here. Backends must initialize all of
    # these in __init__.
    model: str
    workspace: Path
    home_workspace: Path
    max_budget_usd: float
    timeout_seconds: int
    max_context_window: int
    workspace_config: WorkspaceConfig | None
    provider: str

    @abstractmethod
    async def send(self, prompt: str | list, chat_id: int | None = None) -> AsyncIterator[StreamEvent]:
        """Send a message and yield streaming events.

        Implementations should be async generators (use yield, not
        return). The AsyncIterator return type is the consumer-facing
        interface; pool.py iterates over the result, it does not send
        values into the generator.
        """
        ...

    @abstractmethod
    async def change_workspace(self, new_workspace: Path, workspace_config: WorkspaceConfig | None = None) -> None:
        """Switch working directory. Kills current process; next send() restarts."""
        ...

    @abstractmethod
    async def restart(self) -> None:
        """Kill process; next send() starts fresh."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Graceful shutdown with save-prompt."""
        ...

    @abstractmethod
    def force_kill(self) -> None:
        """Immediate kill, no cleanup. Safe without lock."""
        ...

    @property
    @abstractmethod
    def is_alive(self) -> bool:
        """True if the subprocess is running."""
        ...

    @property
    @abstractmethod
    def session_id(self) -> str | None:
        """Current session identifier."""
        ...


# ── Context injection functions ─────────────────────────────────────


def build_session_context(
    *,
    workspace: Path,
    home_workspace: Path,
    api: ApiContext,
    workspace_config: WorkspaceConfig | None,
    chat_id: int | None,
    data_dir: Path,
    memory_enabled: bool = False,
) -> str:
    """
    Build the context prefix for the first message of a new session.

    Always returns a non-empty string because the memory section
    unconditionally appends to parts (exists, empty, or missing).
    The caller prepends this to the prompt before sending.

    Extracted from claude.py _send_locked() lines 414-548.

    Args:
        workspace: The backend's current working directory.
        home_workspace: Kai's home workspace (identity + memory source).
        api: Webhook port, secret, and services info.
        workspace_config: Per-workspace config (for system_prompt).
        chat_id: Telegram chat ID for history scoping and API routing.
        data_dir: Root data directory (memory, history, files live here).
        memory_enabled: Operator-intent flag (Config.memory_enabled).
            Drives the memory subsystem marker emission and gates
            MEMORY.md injection. Reflects intent, not runtime success;
            the runtime-success surface is `memory.is_enabled()`. Passed
            as a bool rather than the full Config to match the existing
            convention of backends taking individual config fields at
            construction.
    """
    parts: list[str] = []

    # When in a foreign workspace, inject Kai's identity from home.
    # try/except guards against race (file deleted between exists()
    # and read_text()) and permission errors, matching the pattern
    # in get_workspace_system_prompt().
    if workspace != home_workspace:
        identity_path = home_workspace / ".claude" / "CLAUDE.md"
        try:
            identity = identity_path.read_text().strip()
            if identity:
                parts.append(f"[Your core identity and instructions:]\n{identity}")
        except OSError:
            pass

    # Memory subsystem state marker. Tells the inner Claude where to
    # route new fact saves: enabled = POST /api/memory/add (Qdrant),
    # disabled = Edit MEMORY.md. Reflects operator INTENT (Config.
    # memory_enabled), not runtime success; if Qdrant init failed at
    # startup, the marker still says "enabled" so write attempts
    # surface the failure as 503s rather than silently re-routing to
    # MEMORY.md. Always emit (per-deployment, not per-user) so the
    # routing rule in CLAUDE.md / PREFERENCES.md can branch on a
    # uniformly-present signal.
    mode = "enabled" if memory_enabled else "disabled"
    parts.append(f"[Memory subsystem: {mode}]")

    # Always inject the per-user PREFERENCES.md as the always-on rule
    # surface. Distinct from MEMORY.md, which is the per-user fact
    # surface; PREFERENCES.md holds operator-personal directives that
    # need to fire on every turn (writing style, formatting, behavioral
    # rules), while MEMORY.md holds project state and notes that surface
    # via similarity retrieval once the Qdrant-backed semantic memory
    # is enabled. Injected above MEMORY.md so the inner Claude reads
    # rules before facts on a top-to-bottom scan. When chat_id is None
    # (one-shot CLI invocations) the block is omitted entirely; there
    # is no global-fallback PREFERENCES.md.
    if chat_id is not None:
        pref_path = data_dir / "preferences" / str(chat_id) / "PREFERENCES.md"
        try:
            pref_text = pref_path.read_text().strip()
            if pref_text:
                parts.append(f"[Your personal preferences (file: {pref_path}):]\n{pref_text}")
            else:
                parts.append(f"[Your personal preferences (file: {pref_path}):]\n(currently empty)")
        except OSError:
            # OSError fires only on missing or unreadable files; the
            # empty case is handled by the else branch above with a
            # distinct "(currently empty)" placeholder. Match the
            # MEMORY.md branch's wording for symmetry.
            parts.append(f"[Your personal preferences (file: {pref_path}):]\n(not yet created)")

    # Inject Kai's personal memory from DATA_DIR ONLY in disabled mode.
    # In enabled mode, Qdrant is the active fact surface (retrieved via
    # memory.format_context in claude.py), and MEMORY.md is dormant;
    # injecting it would create a dual-source collision where retrieval
    # results and MEMORY.md content diverge over time. Gate on operator
    # INTENT (config.memory_enabled), not runtime success: if intent is
    # "Qdrant is the surface" and init failed, do not re-show MEMORY.md.
    #
    # The file lives outside the install tree (/var/lib/kai/memory/ in
    # production) so it survives make install. try/except guards against
    # race and permission errors (same pattern as identity).
    #
    # Per-user scoping (#347): when chat_id is set, the file lives under
    # memory/<chat_id>/MEMORY.md so each user has their own writable
    # copy. The inner Claude subprocess runs as that user's os_user
    # (via sudo -H -u), so ownership of the subdirectory is set to
    # match. A non-service user cannot write the legacy single-global
    # file, which was the bug this scoping fixes. Falls back to the
    # legacy global path when chat_id is None (local-dev backends
    # with no multi-user config) so nothing regresses on single-user
    # setups that never hit the multi-user pool path.
    if not memory_enabled:
        if chat_id is not None:
            memory_path = data_dir / "memory" / str(chat_id) / "MEMORY.md"
        else:
            memory_path = data_dir / "memory" / "MEMORY.md"
        try:
            memory = memory_path.read_text().strip()
            if memory:
                parts.append(f"[Your persistent memory (file: {memory_path}):]\n{memory}")
            else:
                parts.append(f"[Your persistent memory (file: {memory_path}):]\n(currently empty)")
        except OSError:
            parts.append(f"[Your persistent memory (file: {memory_path}):]\n(not yet created)")

    # Per-workspace system prompt from workspaces.yaml. Injected
    # between the identity/memory block and conversation history,
    # so it acts as workspace-specific instructions.
    ws_prompt = get_workspace_system_prompt(workspace_config)
    if ws_prompt:
        parts.append(f"## Workspace Instructions\n\n{ws_prompt}")

    # Always inject the per-user history directory path so the inner
    # Claude's grep/jq searches are naturally scoped to this user.
    history_dir = str(data_dir / "history" / str(chat_id)) if chat_id is not None else str(data_dir / "history")

    # Inject recent conversation history for continuity.
    # Filter by chat_id so each user's session only sees their
    # own messages (Phase 2 per-user data isolation).
    recent = get_recent_history(chat_id=chat_id)
    if recent:
        parts.append(f"[Recent conversations (search {history_dir}/ for full logs):]\n{recent}")
    else:
        parts.append(
            f"[Chat history is stored in {history_dir}/ as daily JSONL files. Search with grep or jq when asked about past conversations.]"
        )

    # Inject scheduling API info (always, so cron works from any workspace).
    # The secret is passed via $KAI_WEBHOOK_SECRET env var (not embedded
    # in prompt text) to prevent leakage through session logs.
    if api.webhook_secret:
        api_note = (
            f"[Scheduling API: To create jobs, use curl (NEVER WebFetch) to POST JSON to "
            f"http://localhost:{api.webhook_port}/api/schedule "
            f"with header 'X-Webhook-Secret: $KAI_WEBHOOK_SECRET' (environment variable). "
            f"Required fields: name, prompt, schedule_type, schedule_data. "
            f"Optional: job_type (reminder|claude), auto_remove (bool). "
            f"To list jobs: GET /api/jobs. To update: PATCH /api/jobs/{{id}}. "
            f"To delete: DELETE /api/jobs/{{id}}.]"
        )
        if workspace != home_workspace:
            api_note = (
                f"[Workspace context: You are working in {workspace}. "
                f"Your home workspace is {home_workspace}.]\n{api_note}"
            )
        parts.append(api_note)

    # Inject messaging and file exchange API info so Claude can
    # proactively send text or files to the user (e.g., when a
    # background task completes or a scheduled job has results).
    if api.webhook_secret:
        parts.append(
            f"[Messaging API: To send a text message to the user proactively "
            f"(e.g., background task results), use curl (NEVER WebFetch) to POST JSON to "
            f"http://localhost:{api.webhook_port}/api/send-message "
            f"with header 'X-Webhook-Secret: $KAI_WEBHOOK_SECRET' (environment variable). "
            f'Required: "text" (the message content). '
            f"Long messages are automatically split at Telegram's 4096-char limit.]"
        )
        files_path = f"{data_dir}/files/{chat_id}/" if chat_id else f"{data_dir}/files/"
        parts.append(
            f"[File API: To send a file to the user, use curl (NEVER WebFetch) to POST JSON to "
            f"http://localhost:{api.webhook_port}/api/send-file "
            f"with header 'X-Webhook-Secret: $KAI_WEBHOOK_SECRET' (environment variable). "
            f'Required: "path" (absolute file path within the current workspace {workspace}). '
            f'Optional: "caption". Images are sent as photos, '
            f"everything else as documents.\n"
            f"Incoming files from the user are auto-saved to "
            f"{files_path} and their paths are included in the message.]"
        )

    # Inject available external services info (only if services are configured)
    if api.services_info and api.webhook_secret:
        svc_lines = [
            "[External Services: To call external APIs, use curl (NEVER WebFetch) to POST JSON to "
            f"http://localhost:{api.webhook_port}/api/services/{{name}} "
            f"with header 'X-Webhook-Secret: $KAI_WEBHOOK_SECRET' (environment variable). "
            "Request JSON fields (all optional): "
            '"body" (dict - forwarded as JSON), '
            '"params" (dict - query parameters), '
            '"path_suffix" (str - appended to base URL).',
            "",
            "Available services:",
        ]
        for svc in api.services_info:
            svc_lines.append(f"  - {svc['name']} ({svc['method']}): {svc['description']}")
            if svc.get("notes"):
                svc_lines.append(f"    Notes: {svc['notes']}")
        svc_lines.append("")
        svc_lines.append(
            "Example (Perplexity web search):\n"
            f"  curl -s -X POST http://localhost:{api.webhook_port}/api/services/perplexity "
            f"-H 'Content-Type: application/json' "
            f"""-H "X-Webhook-Secret: $KAI_WEBHOOK_SECRET" """
            """-d '{"body": {"model": "sonar", "messages": [{"role": "user", "content": "your query"}]}}'"""
        )
        svc_lines.append(
            "Prefer external services over built-in WebSearch/WebFetch when available - they provide better results.]"
        )
        parts.append("\n".join(svc_lines))

    # Include chat_id so inner Claude can pass it back in API
    # calls for correct multi-user routing. Without this, all
    # API calls route to the default admin user.
    if chat_id is not None:
        parts.append(
            f"[Your chat_id for API calls: {chat_id}. Include "
            f'"chat_id": {chat_id} in the JSON body of all '
            f"POST requests to /api/schedule, /api/send-message, "
            f"and /api/send-file so responses route to the "
            f"correct user.]"
        )

    # No trailing \n\n here - prepend_to_prompt() adds the separator
    # between the context block and the user's message.
    return "\n\n".join(parts)


def ensure_user_memory(chat_id: int | None, data_dir: Path) -> None:
    """
    Lazily create the per-user MEMORY.md directory and seed file.

    Idempotent and cheap: a stat, a possible mkdir, a possible copy2.
    Called on every send() path before build_session_context() so that
    a user without a pre-created `memory/<chat_id>/` (single-user dev,
    test runs, or any deployment where the inner Claude runs as the
    same identity as the bot) still gets a writable memory surface on
    their first message.

    Scope of what this function fixes - read carefully:

    * Production multi-user (users.yaml entry has an explicit os_user
      different from the service user): the install step (install.py
      `_apply_migrate`) pre-creates and chowns `memory/<chat_id>/` to
      the user's os_user. mkdir here is then a no-op (the dir already
      exists), and the seed file is already in place. Good.
    * A user added to users.yaml AFTER install with a distinct os_user:
      lazy bootstrap is NOT enough. mkdir here runs as the bot/service
      identity, so the new dir and file are service-owned. The inner
      subprocess (sudo -H -u <os_user>) can read but not write them -
      the same #347 regression. A reinstall (or a manual chown) is
      required for that case.
    * Dev / single-user / users without a distinct os_user: lazy
      bootstrap does the whole job - mkdir + seed copy - inheriting
      the caller's ownership, which IS the writer. Good.

    When chat_id is None (legacy/dev path - no users.yaml, no real
    Telegram session), the function ensures `data_dir/memory/MEMORY.md`
    exists with the same seed-or-placeholder semantics. This mirrors
    the removed main._bootstrap_memory() so the legacy global path
    continues to be writable from a fresh install where memory is
    disabled and users.yaml is absent.

    Silent on OSError: a permissions issue creating per-user memory
    must not prevent the bot from answering a message. The read path
    in build_session_context already returns a placeholder string on
    missing files, so the downstream session still boots cleanly.
    """
    # When chat_id is None we are on the legacy/dev path: there is no
    # users.yaml to drive a per-user subdir, so the file lives at
    # data_dir/memory/MEMORY.md. This mirrors the behavior of the
    # removed main._bootstrap_memory() function so a fresh
    # `python -m kai` (no users.yaml, no prior install, memory disabled)
    # still has a writable memory_root for the inner Claude to update.
    # Without this branch a write attempt from the subprocess would
    # FileNotFoundError on the missing parent directory.
    if chat_id is None:
        user_memory_dir = data_dir / "memory"
        user_memory_file = user_memory_dir / "MEMORY.md"
    else:
        user_memory_dir = data_dir / "memory" / str(chat_id)
        user_memory_file = user_memory_dir / "MEMORY.md"

    try:
        user_memory_dir.mkdir(parents=True, exist_ok=True)
        if not user_memory_file.exists():
            # Seed from the example template if one ships with the
            # source tree. Copy2 preserves mode bits so a deliberately
            # permissive example stays that way.
            example = PROJECT_ROOT / "home" / ".claude" / "MEMORY.md.example"
            if example.is_file():
                shutil.copy2(example, user_memory_file)
            else:
                # No template available (unusual - only happens when the
                # install tree is incomplete). Create a minimal placeholder
                # so the subprocess has a writable file to edit.
                user_memory_file.write_text("# Memory\n")
    except OSError:
        log.warning(
            "ensure_user_memory: could not bootstrap %s",
            user_memory_file,
            exc_info=True,
        )


def ensure_user_preferences(chat_id: int | None, data_dir: Path) -> None:
    """
    Lazily create the per-user PREFERENCES.md directory and seed file.

    Sibling to ensure_user_memory(). Same idempotency, same OSError
    swallow behavior, same multi-user ownership caveats - the install
    step (install.py `_apply_migrate`) pre-creates `preferences/<chat_id>/`
    for every entry in users.yaml and chowns it to that user's os_user
    when one is set; lazy bootstrap is the runtime fallback for chat_ids
    added between installs (a reinstall picks up the ownership in the
    `-- PREFERENCES.md directory ownership --` pass).

    When chat_id is None we are on a one-shot CLI / test path: there
    is no per-user PREFERENCES.md to seed because the inject path
    (build_session_context) skips the block entirely when chat_id is
    None. Return immediately; no global-fallback equivalent.

    Silent on OSError: a permissions issue creating per-user
    preferences must not prevent the bot from answering a message.
    The read path in build_session_context already returns a
    placeholder string on missing files, so the downstream session
    still boots cleanly.
    """
    # No global-fallback path for PREFERENCES.md. The inject branch in
    # build_session_context omits the block entirely when chat_id is
    # None, so there is nothing to seed.
    if chat_id is None:
        return

    user_pref_dir = data_dir / "preferences" / str(chat_id)
    user_pref_file = user_pref_dir / "PREFERENCES.md"

    try:
        user_pref_dir.mkdir(parents=True, exist_ok=True)
        if not user_pref_file.exists():
            # Seed from the example template if one ships with the
            # source tree. Copy2 preserves mode bits so a deliberately
            # permissive example stays that way. Mirrors
            # ensure_user_memory's seed step.
            example = PROJECT_ROOT / "home" / ".claude" / "PREFERENCES.md.example"
            if example.is_file():
                shutil.copy2(example, user_pref_file)
            else:
                # No template available (unusual - only happens when
                # the install tree is incomplete). Create a minimal
                # placeholder so the subprocess has a writable file
                # to edit. Matches ensure_user_memory's `# Memory\n`
                # precedent for symmetry.
                user_pref_file.write_text("# Preferences\n")
    except OSError:
        log.warning(
            "ensure_user_preferences: could not bootstrap %s",
            user_pref_file,
            exc_info=True,
        )


def ensure_user_home(chat_id: int | None, data_dir: Path) -> Path:
    """
    Lazily create the per-user home workspace directory.

    Returns the path that should be used as the user's home workspace
    when users.yaml does not set an explicit home_workspace override.
    Idempotent and cheap: a stat and a possible mkdir. Called on every
    session init (and on `/workspace home`) so that any user without a
    pre-created `home/<chat_id>/` - single-user dev, test runs, or a
    user added to users.yaml after install - still lands in a valid
    directory on their first message.

    This is the companion to ensure_user_memory() above. Same scope
    caveats apply (copied verbatim because the multi-user ownership
    semantics are identical):

    * Production multi-user (users.yaml entry has an explicit os_user
      different from the service user): the install step (install.py
      `_apply_migrate`) pre-creates and chowns `home/<chat_id>/` to the
      user's os_user. mkdir here is then a no-op and the directory is
      already writable by the subprocess identity.
    * A user added to users.yaml AFTER install with a distinct os_user:
      lazy bootstrap is NOT enough. mkdir here runs as the bot/service
      identity, so the new directory is service-owned. The inner
      subprocess (sudo -H -u <os_user>) can read but not write - the
      same #347 regression that memory hit. A reinstall (or a manual
      chown) is required for that case.
    * Dev / single-user / users without a distinct os_user: lazy
      bootstrap does the whole job. mkdir inherits the caller's
      ownership, which IS the subprocess identity.

    When chat_id is None (admin-less startup paths: tests, webhook
    health checks, dev smoke tests) the function returns a fixed
    `data_dir/home/anon` path. That path is only used when there is no
    real user; it is not shared between actual Telegram users because
    any real message carries a chat_id.

    Mode 0755 with per-user chown at install time is what isolates
    users from each other. Matches memory/ exactly (install.py
    `_apply_migrate` uses the same ownership rules). Isolation comes
    from ownership, not mode bits.

    Unlike ensure_user_memory this does NOT seed a template file. The
    home workspace is an empty directory that the user populates
    themselves (cloning a repo, dropping notes, etc.).
    """
    # chat_id of None means we have no real Telegram session to key on
    # (test runs, health checks, smoke runs without users.yaml). Use a
    # fixed "anon" subdir so the resolver always returns a concrete
    # path rather than None; avoids defensive None-checks at every call
    # site.
    if chat_id is None:
        path = data_dir / "home" / "anon"
    else:
        path = data_dir / "home" / str(chat_id)

    # mkdir parents=True is load-bearing: on a brand-new install the
    # data_dir/home/ root may not exist yet (the installer creates it,
    # but dev paths and the tests go straight through lazy bootstrap).
    # exist_ok=True means this is safe to call on every session init.
    path.mkdir(parents=True, exist_ok=True, mode=0o755)
    # mkdir's mode= argument is masked by the process umask, so on a
    # service configured with umask 0o027 the directory can end up 0o750
    # instead of 0o755. That blocks group traversal and can break the
    # inner subprocess when sudo -u targets a different identity. Force
    # the intended bits explicitly - this is the same pattern install.py
    # `_apply_migrate` uses after its mkdir calls. Idempotent on reuse.
    os.chmod(path, 0o755)
    return path


def resolve_home_workspace(
    chat_id: int | None,
    config: Config,
    data_dir: Path | None = None,
) -> Path:
    """
    Return the user's home workspace path.

    Resolution order:
    1. `user.home_workspace` from users.yaml when set (admin override,
       returned as-is with no second-guessing).
    2. `<data_dir>/home/<chat_id>/`, auto-created via ensure_user_home().

    All call sites (pool.py session init / get_workspace fallback,
    bot.py `/workspace home` handler and workspace listings) MUST go
    through this function rather than reading any global home field on
    Config directly - that field was removed by issue #353. The old
    global fallback (PROJECT_ROOT / "home" / shared "/opt/kai/home/")
    was a multi-user privacy hazard (every user landed in a directory
    every other user could read).

    Passing `config` (not just its admin home field) keeps the
    resolution decision local: the caller does not need to know whether
    to prefer users.yaml or the per-user default.

    The `data_dir` parameter is exposed for tests that want to thread a
    tmp_path in cleanly (beats monkeypatching the module attribute,
    because a test that forgets the patch would silently write to the
    real `/var/lib/kai`). When None, we resolve from the module-level
    DATA_DIR at CALL time (not definition time): a `data_dir=DATA_DIR`
    default would freeze the value at import, defeating the
    `monkeypatch.setattr("kai.backend.DATA_DIR", ...)` pattern that
    pool.py and bot.py tests still depend on. The underlying
    ensure_user_home already takes data_dir as a parameter; this
    mirrors that signature.
    """
    # Admin-authored override from users.yaml wins outright. No check
    # that the path exists - the config-load validator (config.py
    # load_config) already filtered out bad paths, and a path that
    # became invalid post-load (unmounted drive, etc.) is the admin's
    # problem to fix, not ours to paper over.
    user = config.get_user_config(chat_id) if chat_id is not None else None
    if user and user.home_workspace:
        return user.home_workspace

    # No users.yaml override: use the per-user Kai-managed directory
    # under data_dir. Resolve DATA_DIR from the module here (not as a
    # default arg value) so monkeypatching `kai.backend.DATA_DIR` in
    # tests still flows through. ensure_user_home is idempotent so
    # calling it on every resolution is cheap.
    if data_dir is None:
        data_dir = DATA_DIR
    return ensure_user_home(chat_id, data_dir)


def build_foreign_workspace_reminder(workspace: Path, home_workspace: Path) -> str | None:
    """
    Build the per-message reminder for foreign workspaces.

    Returns the reminder string if workspace != home_workspace, else
    None. Applied on EVERY message, not just the first.

    Extracted from claude.py _send_locked() lines 557-569.
    """
    if workspace == home_workspace:
        return None
    return (
        "[IMPORTANT: This message is from a user via Telegram. "
        "Respond ONLY to what they wrote below. Do NOT continue, "
        "resume, or start any previous work, plans, or tasks.]"
    )


def prepend_to_prompt(prompt: str | list, prefix: str) -> str | list:
    """
    Prepend a text prefix to a prompt.

    Handles both str prompts and list-of-content-blocks prompts.
    No-ops if prefix is empty.
    """
    if not prefix:
        return prompt
    if isinstance(prompt, str):
        # String prompts get a \n\n separator between prefix and prompt.
        return prefix + "\n\n" + prompt
    # List prompts: insert a separate text block at the front. No \n\n
    # needed because content blocks are independent elements in the API.
    return [{"type": "text", "text": prefix}] + prompt


def get_workspace_system_prompt(
    workspace_config: WorkspaceConfig | None,
) -> str | None:
    """
    Read the system prompt for a workspace config.

    Returns the inline system_prompt if set, reads from
    system_prompt_file on each call to pick up changes, or returns
    None if neither is configured.
    """
    if not workspace_config:
        return None
    if workspace_config.system_prompt:
        return workspace_config.system_prompt
    if workspace_config.system_prompt_file:
        # File path was validated at config load time (fail-fast on typos).
        # Read content here so updates are picked up without restart.
        try:
            return workspace_config.system_prompt_file.read_text()
        except OSError:
            log.warning("Cannot read system_prompt_file: %s", workspace_config.system_prompt_file)
            return None
    return None
