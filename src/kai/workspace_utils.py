"""
Workspace path utilities shared between bot.py and pool.py.

Extracted to break the circular import that prevented pool.py
from importing workspace helpers directly from bot.py (bot
imports pool, so pool cannot import bot). See #223 finding 1.
"""

from pathlib import Path


def is_workspace_allowed(path: Path, base: Path | None, allowed: list[Path]) -> bool:
    """Return True if path is covered by a configured workspace source.

    Accepts paths under the user's workspace_base or in their allowed
    list. If neither source is configured, all paths are accepted
    (permissive mode for installs that don't restrict workspace access).

    Args:
        path: The workspace path to validate (need not exist).
        base: The user's resolved workspace_base, or None.
        allowed: The user's effective allowed workspace list
            (pre-resolved by resolve_workspace_access).
    """
    if not base and not allowed:
        # No restrictions configured - open access
        return True
    resolved = path.resolve()
    # Resolve base too so symlinks in the base path don't bypass the check
    resolved_base = base.resolve() if base else None
    in_base = resolved_base and resolved.is_relative_to(resolved_base)
    # allowed list is pre-resolved by resolve_workspace_access(),
    # so no need to call .resolve() again on each entry.
    in_allowed = resolved in allowed
    return bool(in_base or in_allowed)
