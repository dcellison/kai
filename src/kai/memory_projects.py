"""
Memory project detection: map a workspace path to its registered
memory project, if any.

The registry itself lives on `Config.memory_projects` and is loaded
from `memory-projects.yaml` by `kai.config._load_memory_project_configs`.
This module is the read-only consumer: it accepts a path and a
registry dict and returns the matching `ActiveMemoryProject`, or
`None` when the path is not registered.

The detector is intentionally pure: it does not touch process state
(no `Path.cwd()`, no globals), so callers must pass the active
workspace path explicitly. Runtime callers should pass the backend's
active `workspace` path (the chat/session workspace tracked per user
by `kai.pool`); tests can pass any path constructed against
`tmp_path`.

Detection rules (also documented inline below):

1. Resolve the input path with `expanduser().resolve()`.
2. A registry root matches when the resolved path equals the root
   or `Path.is_relative_to()` the root.
3. When multiple roots match, the root with the most path parts
   wins (longest-prefix). Nested project roots are common
   (`/Users/kai/Projects/kai` vs `/Users/kai/Projects`), so the
   narrower root must take precedence.
4. Unknown paths return `None`. Detection does not guess.
5. Disabled projects (`memory_enabled=False`) still return an
   `ActiveMemoryProject` so logs and future shadow-mode metrics can
   distinguish "no project here" from "known project with memory
   disabled". Retrieval and write code must treat
   `memory_enabled=False` as "global-only behavior".

The detector does NOT change retrieval, prompt rendering, or write
routing. Wiring those paths to the detector belongs to #544 and
later issues in the scoped-memory epic.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from weakref import WeakKeyDictionary

from kai.config import MemoryProjectConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActiveMemoryProject:
    """
    The memory project that a given workspace path belongs to.

    Returned by `detect_active_memory_project()` when the input path
    matches a registered root. Carries the matched root so callers
    can log which root won (useful when nested roots are configured)
    and the policy field that future write-routing will consume.

    Attributes:
        project_id: The registry key of the matched project.
        display_name: Human-readable name for logs and operator UI.
        matched_root: The specific resolved registry root that won
            the longest-prefix match. Useful for diagnostics when
            multiple roots could have matched.
        memory_enabled: When False, the project is detectable but
            project-scoped retrieval and writes are not allowed;
            callers should fall back to global-only behavior.
        default_scope_for_new_facts: Optional policy hint for the
            future write-scope routing path. Inert in #543; later
            extraction code may consult it.
    """

    project_id: str
    display_name: str
    matched_root: Path
    memory_enabled: bool
    default_scope_for_new_facts: str | None


def detect_active_memory_project(
    cwd: Path,
    projects: dict[str, MemoryProjectConfig],
) -> ActiveMemoryProject | None:
    """
    Return the memory project that owns `cwd`, or None.

    The path is resolved (`expanduser().resolve()`) before matching
    so that symlink aliases collapse to the same canonical location
    and a project cannot be "two projects" depending on which alias
    the operator used to enter it.

    Matching uses `Path.is_relative_to()` (and equality), NOT
    string-prefix comparison: `/work/foo2` must not match a root of
    `/work/foo`. When multiple roots match, the root with the most
    path parts wins.

    Args:
        cwd: The workspace path to detect from. Runtime callers
            should pass the backend's active workspace path;
            tests can pass any path.
        projects: The memory project registry from
            `Config.memory_projects`. An empty dict returns None
            for every input.

    Returns:
        `ActiveMemoryProject` when `cwd` is inside (or equals) a
        registered root. `None` when no root matches; the caller
        should fall back to global-only behavior.
    """
    if not projects:
        return None

    try:
        resolved_cwd = Path(cwd).expanduser().resolve()
    except (OSError, ValueError) as e:
        log.warning("detect_active_memory_project: cannot resolve cwd %r: %s", cwd, e)
        return None

    # Track the best match by (path-parts depth, project, root). The
    # registry loader has already deduplicated roots across projects,
    # so two registered roots with the same depth covering the same
    # cwd is structurally impossible: equality at the same depth
    # would mean identical roots, which the loader rejects.
    best_depth = -1
    best_project: MemoryProjectConfig | None = None
    best_root: Path | None = None

    for project in projects.values():
        for root in project.workspace_roots:
            # Path containment, not string prefix. is_relative_to
            # returns False for equality, so handle equality as a
            # separate match.
            if resolved_cwd != root and not resolved_cwd.is_relative_to(root):
                continue
            depth = len(root.parts)
            if depth > best_depth:
                best_depth = depth
                best_project = project
                best_root = root

    if best_project is None or best_root is None:
        return None

    return ActiveMemoryProject(
        project_id=best_project.project_id,
        display_name=best_project.display_name,
        matched_root=best_root,
        memory_enabled=best_project.memory_enabled,
        default_scope_for_new_facts=best_project.default_scope_for_new_facts,
    )


# ── User-registered registry (DB layer) ─────────────────────────────
# Chat-registered projects live in the session DB (kai.sessions owns
# persistence) and in this module-level cache (this module owns the
# merged detection view). The daemon is the only writer: command
# handlers mutate the DB and this cache in the same turn, so the
# cache never goes stale and detection sees changes on the next
# message with no restart. All mutation happens on the event loop,
# never in an executor, so no locking is needed.

# Mirrors the canonical SCOPE_GLOBAL / SCOPE_PROJECT constants in
# kai.memory. Duplicated as literals so this pure-detection module's
# import surface stays at kai.config only; importing kai.memory here
# would pull the whole storage stack into every consumer of the
# detector. The YAML loader makes the same trade with its own lazy
# import of the constants.
_VALID_DEFAULT_SCOPES: frozenset[str] = frozenset({"global", "project"})

_db_registry: dict[str, MemoryProjectConfig] = {}
_db_creators: dict[str, int] = {}

# Serializes registry mutations (guard + DB write + cache update).
# The nested-root and collision guards are reads of the merged view
# taken BEFORE an awaited DB insert; under concurrent command
# handling, two registrations can both pass their guards against the
# same stale view and commit a parent/child pair that the guards
# exist to prevent (the child then steals its parent's subtree via
# longest-prefix detection). The DB's uniqueness constraints only
# cover exact id/root equality, not containment, so the stable-view
# property must come from serialization. Mutations are rare,
# operator-driven events; a single registry-wide lock costs nothing.
#
# Per-loop factory rather than a module-level Lock: asyncio locks
# bind to the event loop that first acquires them and raise when
# acquired from another loop. The daemon runs one loop forever, so
# production sees a single lock; the factory exists so any context
# with a different loop lifecycle (tests, future tooling) gets a
# lock bound to ITS loop instead of a cross-loop RuntimeError.
_registry_mutation_locks: WeakKeyDictionary = WeakKeyDictionary()


def registry_mutation_lock() -> asyncio.Lock:
    """The registry mutation lock for the running event loop."""
    loop = asyncio.get_running_loop()
    lock = _registry_mutation_locks.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _registry_mutation_locks[loop] = lock
    return lock


def _row_to_config(row: dict) -> MemoryProjectConfig | None:
    """
    Validate one DB row into a MemoryProjectConfig, or None.

    Applies the same per-entry rules as the YAML loader in
    kai.config (non-empty id and display name, real boolean
    memory_enabled, recognized default scope, resolvable root) so a
    row that would have been skipped as a YAML entry is skipped as a
    DB row too. Fail-closed: a malformed row never reaches
    detection.
    """
    project_id = row.get("project_id")
    display_name = row.get("display_name")
    if not isinstance(project_id, str) or not project_id.strip():
        log.warning("memory_projects db row skipped: bad project_id %r", project_id)
        return None
    if not isinstance(display_name, str) or not display_name.strip():
        log.warning("memory_projects db row %r skipped: bad display_name", project_id)
        return None
    memory_enabled = row.get("memory_enabled")
    if not isinstance(memory_enabled, bool):
        log.warning("memory_projects db row %r skipped: bad memory_enabled %r", project_id, memory_enabled)
        return None
    default_scope = row.get("default_scope_for_new_facts")
    if default_scope is not None and default_scope not in _VALID_DEFAULT_SCOPES:
        log.warning("memory_projects db row %r skipped: bad default scope %r", project_id, default_scope)
        return None
    raw_root = row.get("workspace_root")
    if not isinstance(raw_root, str) or not raw_root.strip():
        log.warning("memory_projects db row %r skipped: bad workspace_root %r", project_id, raw_root)
        return None
    try:
        resolved = Path(raw_root).expanduser().resolve()
    except (OSError, ValueError) as e:
        log.warning("memory_projects db row %r skipped: cannot resolve root %r: %s", project_id, raw_root, e)
        return None
    return MemoryProjectConfig(
        project_id=project_id.strip(),
        display_name=display_name.strip(),
        workspace_roots=(resolved,),
        memory_enabled=memory_enabled,
        default_scope_for_new_facts=default_scope,
    )


def load_db_registry(rows: list[dict]) -> None:
    """
    Replace the cache from persisted rows (startup load).

    Intra-DB duplicates cannot exist (PRIMARY KEY and UNIQUE root
    constraints), so this is a straight validated load; conflicts
    with the YAML layer are handled per-lookup in merged_registry,
    not here, so an operator pinning a YAML entry over an existing
    DB row needs no DB cleanup.
    """
    _db_registry.clear()
    _db_creators.clear()
    for row in rows:
        cfg = _row_to_config(row)
        if cfg is None:
            continue
        _db_registry[cfg.project_id] = cfg
        created_by = row.get("created_by")
        if isinstance(created_by, int):
            _db_creators[cfg.project_id] = created_by


def db_registry_upsert(row: dict) -> bool:
    """Add one registered project to the cache (post-DB-write hook
    for the register handler). Returns False when validation rejects
    the row; the handler treats that as an internal error since it
    validated the inputs before writing."""
    cfg = _row_to_config(row)
    if cfg is None:
        return False
    _db_registry[cfg.project_id] = cfg
    created_by = row.get("created_by")
    if isinstance(created_by, int):
        _db_creators[cfg.project_id] = created_by
    return True


def db_registry_remove(project_id: str) -> None:
    """Drop one registered project from the cache (post-DB-delete
    hook for the unregister handler)."""
    _db_registry.pop(project_id, None)
    _db_creators.pop(project_id, None)


def db_registry_creator(project_id: str) -> int | None:
    """chat_id that registered a project, for the unregister
    permission check. None for YAML-pinned or unknown projects."""
    return _db_creators.get(project_id)


def merged_registry(yaml_projects: dict[str, MemoryProjectConfig]) -> dict[str, MemoryProjectConfig]:
    """
    The detection view: user-registered projects under the
    operator-pinned YAML layer.

    YAML wins every conflict, on project_id AND on workspace-root
    ownership, mirroring the YAML loader's own first-wins duplicate
    rules; a user registration can never shadow or steal scope from
    an operator-pinned project. Returns the yaml dict unchanged when
    no DB rows exist, so the pre-registration call sites behave
    byte-identically.
    """
    if not _db_registry:
        return yaml_projects
    yaml_roots = {root for project in yaml_projects.values() for root in project.workspace_roots}

    def _yaml_owned(root: Path) -> bool:
        # Equality OR containment, matching the detector's own
        # containment model: a DB root INSIDE a YAML root would win
        # that subtree via longest-prefix matching, so a later
        # operator pin of the parent must evict the persisted child,
        # not coexist with it. The inverse (a YAML root inside a DB
        # root) is ordinary nested-project coexistence: the deeper
        # YAML root already owns its subtree through longest-prefix,
        # exactly like nested YAML entries.
        return any(root == yaml_root or root.is_relative_to(yaml_root) for yaml_root in yaml_roots)

    merged: dict[str, MemoryProjectConfig] = {}
    for project_id, cfg in _db_registry.items():
        if project_id in yaml_projects:
            log.debug("memory_projects: db project %r shadowed by yaml id", project_id)
            continue
        if any(_yaml_owned(root) for root in cfg.workspace_roots):
            log.debug("memory_projects: db project %r root owned by yaml layer", project_id)
            continue
        merged[project_id] = cfg
    merged.update(yaml_projects)
    return merged
