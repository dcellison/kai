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

import logging
from dataclasses import dataclass
from pathlib import Path

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
