"""
Every direct vector-store write or delete outside `kai.memory` is a reviewed exception.

On a protected install, current truth lives in the canonical fact and
episode lifecycle; the vector store is only its projection. A module that
writes or deletes vector rows directly bypasses that authority: a direct
write creates a row the current-truth gate never recalls once the owner's
legacy memory is reconciled, and a direct delete leaves a claim active
with a projection pointing at nothing.

This test lists every production call site of the direct primitives with
the reason it is allowed, and fails on any site not in the list and on any
listed site that no longer exists, so the list stays exact. A new write
path should go through `MemoryFactLifecycleService` or
`MemoryEpisodeHistoryService` instead of being added here.
"""

from __future__ import annotations

import ast
from pathlib import Path

import kai

# The primitives that touch vector rows without the lifecycle.
_PRIMITIVES = frozenset({"add_structured", "delete_by_id", "delete_all"})

# (module path under src/kai, enclosing function, primitive) -> why it is allowed.
_ALLOWED = {
    ("workshop/fact_lifecycle.py", "add", "add_structured"): "the fact lifecycle's own vector adapter",
    ("workshop/episode_history.py", "add", "add_structured"): "the episode lifecycle's own vector adapter",
    ("workshop/memory_queries.py", "delete", "delete_by_id"): "deleting a legacy episode row, which has no lifecycle",
    ("webhook.py", "_handle_memory_add", "add_structured"): "installs without protected canonical authority",
    ("webhook.py", "_handle_memory_delete_all", "delete_all"): "installs without protected canonical authority",
    ("memory_extraction.py", "_store_facts", "add_structured"): "extraction runs without canonical provenance",
    ("memory_extraction.py", "_store_facts", "delete_by_id"): "extraction runs without canonical provenance",
    ("memory_extraction.py", "_generate_episode", "add_structured"): "episode runs without canonical provenance",
    ("memory_scope_review.py", "run_apply", "delete_by_id"): "operator legacy scope review, legacy rows only",
    ("memory_admin.py", "_cmd_purge_sandbox", "delete_all"): "operator purge of a sandbox evaluation user",
}


def _call_sites(root: Path) -> set[tuple[str, str, str]]:
    """Find every reference to a primitive as `memory.<name>` or a bare imported name."""
    sites: set[tuple[str, str, str]] = set()
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        # `kai.memory` defines the primitives; the eval harness writes to
        # sandbox users that never reach a protected owner's recall.
        if relative == "memory.py" or relative.startswith("eval/"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))

        def visit(node: ast.AST, function: str, relative: str = relative) -> None:
            for child in ast.iter_child_nodes(node):
                inner = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else function
                if (
                    isinstance(child, ast.Attribute)
                    and child.attr in _PRIMITIVES
                    and isinstance(child.value, ast.Name)
                    and child.value.id in {"memory", "memory_module"}
                ):
                    sites.add((relative, function, child.attr))
                if isinstance(child, ast.Name) and child.id in _PRIMITIVES and isinstance(child.ctx, ast.Load):
                    sites.add((relative, function, child.id))
                visit(child, inner)

        visit(tree, "<module>")
    return sites


def test_direct_vector_writes_are_exactly_the_reviewed_sites() -> None:
    root = Path(kai.__file__).parent

    sites = _call_sites(root)

    assert sites - set(_ALLOWED) == set(), "route new memory writes through the fact or episode lifecycle"
    assert set(_ALLOWED) - sites == set(), "remove allowlist entries whose call site is gone"
